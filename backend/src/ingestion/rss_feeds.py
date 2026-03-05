"""
RSS Feed Ingestion Module.

WHAT IS RSS?
------------
RSS (Really Simple Syndication) is an XML format that news organizations use to
publish their latest headlines. Each feed is a URL that returns structured data:
headline, link, publication time, and a brief summary.

WHY RSS?
--------
RSS complements GDELT in our intelligence pipeline:
- GDELT tells us WHAT HAPPENED (structured events: "Country A attacked Country B")
- RSS tells us HOW IT'S BEING REPORTED (narrative framing, emphasis, tone)

The framing matters because markets react to narratives, not just facts. A conflict
event reported as "escalating tensions" vs "minor border skirmish" drives different
sentiment and different market moves.

FEEDS:
------
- Reuters World News: Western wire service, fast, factual
- AP News World: Second major Western wire service, broad coverage
- BBC World News: British perspective, strong international desk
- Al Jazeera: Middle Eastern perspective, critical for regional coverage

Source diversity reduces bias — an important concept in all-source intelligence.

USAGE:
------
    from src.ingestion.rss_feeds import RSSIngestion

    rss = RSSIngestion()

    # Fetch headlines without storing
    headlines = rss.fetch_feed("reuters")

    # Fetch and store all feeds
    counts = rss.ingest_all_feeds()
    print(counts)  # {"reuters": 15, "ap": 12, "bbc": 10, "aljazeera": 8}
"""

import logging
from datetime import datetime, timezone

import feedparser
import requests

from src.config.settings import RSS_FEEDS, RSS_REQUEST_TIMEOUT
from src.db.connection import get_session
from src.db.queries import upsert_headline

logger = logging.getLogger(__name__)


class RSSIngestion:
    """
    Handles fetching and storing news headlines from RSS feeds.

    Follows the same pattern as GDELTIngestion:
    1. fetch_* methods retrieve data without storing
    2. ingest_* methods retrieve and store in the database
    3. Deduplication via URL uniqueness (upsert_headline skips existing URLs)
    """

    def __init__(self, feeds: list[dict] | None = None, timeout: int | None = None):
        """
        Initialize the RSS ingestion handler.

        Args:
            feeds: List of feed dicts with "name" and "url" keys.
                   Defaults to RSS_FEEDS from settings.
            timeout: Request timeout in seconds. Defaults to RSS_REQUEST_TIMEOUT.
        """
        self.feeds = feeds or RSS_FEEDS
        self.timeout = timeout or RSS_REQUEST_TIMEOUT

    def _parse_published_date(self, entry: dict) -> datetime | None:
        """
        Extract and parse the publication date from an RSS entry.

        RSS feeds use various date formats. feedparser normalizes most of them
        into a 9-tuple (struct_time), but some entries have no date at all.

        Args:
            entry: A single item from feedparser's parsed feed

        Returns:
            datetime object in UTC, or None if unparseable
        """
        # feedparser provides a normalized time tuple in 'published_parsed'
        parsed_time = entry.get("published_parsed")
        if parsed_time:
            try:
                return datetime(*parsed_time[:6], tzinfo=timezone.utc)
            except (ValueError, TypeError):
                pass

        # Fallback: try 'updated_parsed' (Atom feeds use this)
        updated_time = entry.get("updated_parsed")
        if updated_time:
            try:
                return datetime(*updated_time[:6], tzinfo=timezone.utc)
            except (ValueError, TypeError):
                pass

        return None

    def _clean_headline(self, text: str | None) -> str | None:
        """
        Clean headline text by stripping whitespace and HTML artifacts.

        Args:
            text: Raw headline text from RSS

        Returns:
            Cleaned text, or None if empty
        """
        if not text:
            return None
        cleaned = text.strip()
        # Some feeds include HTML entities or extra whitespace
        cleaned = " ".join(cleaned.split())
        return cleaned if cleaned else None

    def _entry_to_headline_dict(self, entry: dict, source: str) -> dict | None:
        """
        Convert a feedparser entry to a dict matching our NewsHeadline model.

        Mirrors GDELTIngestion._row_to_event_dict() — takes raw data from the
        source and maps it to our database schema.

        Args:
            entry: Single item from feedparser
            source: Feed name ("reuters", "ap", etc.)

        Returns:
            Dict ready for upsert_headline(), or None if entry is unusable
        """
        headline = self._clean_headline(entry.get("title"))
        url = entry.get("link")
        published_at = self._parse_published_date(entry)

        # Skip entries missing required fields
        if not headline or not url:
            return None

        # If no publication date, use current time (better than dropping the headline)
        if not published_at:
            published_at = datetime.now(timezone.utc)

        description = self._clean_headline(entry.get("summary"))

        return {
            "source": source,
            "headline": headline,
            "url": url,
            "description": description,
            "published_at": published_at,
            # sentiment_score and sentiment_label left as NULL — Phase 2
        }

    def fetch_feed(self, feed_name: str) -> list[dict]:
        """
        Fetch and parse headlines from a single RSS feed.

        This is the read-only method — fetches data without storing it.
        Useful for testing or exploration.

        Args:
            feed_name: Internal name of the feed ("reuters", "ap", etc.)

        Returns:
            List of headline dicts ready for database storage

        Raises:
            ValueError: If feed_name is not in the configured feeds
        """
        # Find the feed config by name
        feed_config = next(
            (f for f in self.feeds if f["name"] == feed_name),
            None,
        )
        if not feed_config:
            raise ValueError(
                f"Unknown feed '{feed_name}'. "
                f"Available: {[f['name'] for f in self.feeds]}"
            )

        url = feed_config["url"]
        logger.info(f"Fetching RSS feed: {feed_name} from {url}")

        try:
            # feedparser can fetch URLs directly, but we use requests for
            # timeout control and consistent error handling with our other ingestors
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()

            # Parse the XML content
            parsed = feedparser.parse(response.content)

            if parsed.bozo and not parsed.entries:
                # 'bozo' means feedparser found XML errors.
                # If there are still entries, the data is usable despite errors.
                # If there are NO entries, the feed is truly broken.
                logger.warning(
                    f"Feed '{feed_name}' returned malformed XML with no entries: "
                    f"{parsed.bozo_exception}"
                )
                return []

            # Convert each entry to our schema
            headlines = []
            for entry in parsed.entries:
                headline_dict = self._entry_to_headline_dict(entry, feed_name)
                if headline_dict:
                    headlines.append(headline_dict)

            logger.info(f"Parsed {len(headlines)} headlines from {feed_name}")
            return headlines

        except requests.RequestException as e:
            logger.error(f"Failed to fetch feed '{feed_name}': {e}")
            return []

    def ingest_feed(self, feed_name: str) -> int:
        """
        Fetch a single feed and store headlines in the database.

        Mirrors GDELTIngestion.ingest_date() — fetch + store + return count.

        Args:
            feed_name: Internal name of the feed

        Returns:
            Number of NEW headlines stored (skips duplicates)
        """
        headlines = self.fetch_feed(feed_name)

        if not headlines:
            return 0

        new_count = 0
        with get_session() as session:
            for headline_data in headlines:
                result = upsert_headline(session, headline_data)
                if result is not None:
                    new_count += 1

        logger.info(
            f"Ingested {feed_name}: {new_count} new headlines "
            f"({len(headlines) - new_count} duplicates skipped)"
        )
        return new_count

    def ingest_all_feeds(self) -> dict[str, int]:
        """
        Fetch and store headlines from all configured feeds.

        Mirrors GDELTIngestion.ingest_date_range() — loops through all sources,
        catches per-source errors so one broken feed doesn't stop the rest.

        Returns:
            Dict of {feed_name: count} for each feed. -1 indicates an error.
        """
        results = {}

        for feed_config in self.feeds:
            feed_name = feed_config["name"]
            try:
                count = self.ingest_feed(feed_name)
                results[feed_name] = count
            except Exception as e:
                logger.error(f"Error ingesting feed '{feed_name}': {e}")
                results[feed_name] = -1

        total = sum(c for c in results.values() if c > 0)
        logger.info(f"RSS ingestion complete: {total} new headlines across {len(results)} feeds")
        return results


# Convenience function for quick testing (matches fetch_sample_events pattern)
def fetch_sample_headlines(feed_name: str = "reuters", limit: int = 5) -> list[dict]:
    """
    Fetch a few sample headlines for testing/exploration.

    Args:
        feed_name: Feed to fetch from (default: reuters)
        limit: Number of headlines to return

    Returns:
        List of headline dicts
    """
    rss = RSSIngestion()
    headlines = rss.fetch_feed(feed_name)
    return headlines[:limit]
