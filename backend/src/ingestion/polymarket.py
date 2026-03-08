"""
Polymarket Prediction Market Ingestion Module.

WHAT IS POLYMARKET?
-------------------
Polymarket is a prediction market where people bet real money on the outcomes
of future events. The price of a "Yes" share reflects the crowd's estimated
probability of that outcome. For example, "US x Iran ceasefire by March 31?"
trading at $0.24 means the crowd gives it a ~24% chance.

WHY PREDICTION MARKETS?
-----------------------
Prediction markets complement our other data sources:
- GDELT/RSS tells us what HAPPENED (backward-looking)
- FRED tells us the macro ENVIRONMENT (lagging indicators)
- Polymarket tells us what the crowd EXPECTS to happen (forward-looking)

When a conflict probability spikes before financial markets react, that's
a potential alpha signal. Markets are also useful for calibrating our own
risk assessments — if the crowd says 35% and our model says 80%, one of
us is wrong.

API DETAILS:
------------
- Gamma API: https://gamma-api.polymarket.com (market metadata, public, no auth)
- No API key needed — fully public read access
- Rate limits: 300-500 req/10s (extremely generous for periodic ingestion)
- We fetch /events (which include nested markets) and filter for geopolitical
  topics using tag-based classification

FILTERING STRATEGY:
-------------------
Polymarket has markets on everything (sports, crypto, pop culture). We filter
by checking each event's tags against include/exclude lists:
- Include: geopolitics, world, politics, economy, iran, russia, china, etc.
- Exclude: sports, soccer, nba, crypto, etc.
This is done client-side because the API's tag filter doesn't work reliably.

USAGE:
------
    from src.ingestion.polymarket import PolymarketIngestion

    pm = PolymarketIngestion()

    # Fetch without storing
    markets = pm.fetch_markets()

    # Fetch and store all geopolitical markets
    count = pm.ingest_markets()
    print(f"Stored {count} prediction markets")
"""

import logging
from datetime import datetime, timezone

import requests

from src.config.settings import (
    POLYMARKET_GEOPOLITICAL_TAGS,
    POLYMARKET_EXCLUDE_TAGS,
    POLYMARKET_REQUEST_TIMEOUT,
)
from src.db.connection import get_session
from src.db.queries import upsert_prediction_market

logger = logging.getLogger(__name__)

GAMMA_API_URL = "https://gamma-api.polymarket.com"


class PolymarketIngestion:
    """
    Handles fetching and storing prediction market data from Polymarket.

    Follows the same pattern as RSSIngestion / FREDIngestion:
    1. fetch_* methods retrieve data without storing
    2. ingest_* methods retrieve and store in the database
    3. Deduplication via (market_id, snapshot_date) uniqueness
    """

    def __init__(
        self,
        geo_tags: set[str] | None = None,
        exclude_tags: set[str] | None = None,
        timeout: int | None = None,
    ):
        """
        Initialize the Polymarket ingestion handler.

        Args:
            geo_tags: Tag slugs that indicate a geopolitical market.
            exclude_tags: Tag slugs that indicate non-geopolitical markets.
            timeout: Request timeout in seconds.
        """
        self.geo_tags = geo_tags or POLYMARKET_GEOPOLITICAL_TAGS
        self.exclude_tags = exclude_tags or POLYMARKET_EXCLUDE_TAGS
        self.timeout = timeout or POLYMARKET_REQUEST_TIMEOUT

    def _is_geopolitical(self, event: dict) -> bool:
        """
        Determine if a Polymarket event is geopolitically relevant.

        An event qualifies if it has at least one geopolitical tag
        AND no exclusion tags (sports, crypto, etc.).

        Args:
            event: A Polymarket event dict from the Gamma API

        Returns:
            True if the event is geopolitically relevant
        """
        tag_slugs = {t.get("slug", "").lower() for t in event.get("tags", [])}
        has_geo = bool(tag_slugs & self.geo_tags)
        has_exclude = bool(tag_slugs & self.exclude_tags)
        return has_geo and not has_exclude

    def _parse_outcome_price(self, prices_str: str, index: int = 0) -> float | None:
        """
        Extract a probability from Polymarket's outcomePrices field.

        outcomePrices is a JSON string like '["0.35", "0.65"]' where
        index 0 = "Yes" probability, index 1 = "No" probability.

        Args:
            prices_str: The outcomePrices string from the API
            index: Which outcome to extract (0 = Yes, 1 = No)

        Returns:
            Probability as a float (0.0 to 1.0), or None if unparseable
        """
        import json

        if not prices_str:
            return None
        try:
            prices = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
            if isinstance(prices, list) and len(prices) > index:
                return float(prices[index])
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        return None

    def _market_to_dict(self, market: dict, event_title: str) -> dict | None:
        """
        Convert a Polymarket market to a dict matching our PredictionMarket model.

        Each market is a single Yes/No question within an event. For example,
        the event "Fed decision in March?" might have markets for "decrease 25bps",
        "decrease 50bps", "increase 25bps", etc.

        Args:
            market: A market dict from the Gamma API (nested under an event)
            event_title: The parent event's title, for context

        Returns:
            Dict ready for upsert_prediction_market(), or None if unusable
        """
        question = market.get("question")
        market_id = market.get("id")

        if not question or not market_id:
            return None

        yes_price = self._parse_outcome_price(market.get("outcomePrices"))
        if yes_price is None:
            return None

        return {
            "market_id": str(market_id),
            "question": question,
            "event_title": event_title,
            "yes_price": yes_price,
            "volume_24h": float(market.get("volume24hr", 0) or 0),
            "total_volume": float(market.get("volume", 0) or 0),
            "liquidity": float(market.get("liquidity", 0) or 0),
            "end_date": market.get("endDate"),
            "snapshot_at": datetime.now(timezone.utc),
        }

    def fetch_markets(self, limit: int = 100) -> list[dict]:
        """
        Fetch geopolitical prediction markets from Polymarket.

        Fetches events from the Gamma API, filters for geopolitical relevance,
        and flattens the nested markets into a list of dicts.

        Args:
            limit: Maximum number of events to fetch from the API.
                   The actual number of markets may be higher since
                   each event contains multiple markets.

        Returns:
            List of market dicts ready for database storage.
        """
        logger.info("Fetching Polymarket events")

        try:
            response = requests.get(
                f"{GAMMA_API_URL}/events",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": str(limit),
                    "order": "volume24hr",
                    "ascending": "false",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            events = response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch Polymarket events: {e}")
            return []

        # Filter to geopolitical events and flatten markets
        markets = []
        geo_event_count = 0

        for event in events:
            if not self._is_geopolitical(event):
                continue

            geo_event_count += 1
            event_title = event.get("title", "Unknown")

            for market in event.get("markets", []):
                market_dict = self._market_to_dict(market, event_title)
                if market_dict:
                    markets.append(market_dict)

        logger.info(
            f"Parsed {len(markets)} geopolitical markets "
            f"from {geo_event_count} events "
            f"(filtered from {len(events)} total events)"
        )
        return markets

    def ingest_markets(self, limit: int = 100) -> int:
        """
        Fetch geopolitical markets and store snapshots in the database.

        Each run creates a new snapshot for each market (same market_id,
        different snapshot date). This lets us track how probabilities
        change over time — a key signal for our analysis.

        Args:
            limit: Maximum events to fetch from the API

        Returns:
            Number of market snapshots stored. -1 on error.
        """
        try:
            markets = self.fetch_markets(limit=limit)
        except Exception as e:
            logger.error(f"Error fetching Polymarket data: {e}")
            return -1

        if not markets:
            return 0

        new_count = 0
        with get_session() as session:
            for market_data in markets:
                result = upsert_prediction_market(session, market_data)
                if result is not None:
                    new_count += 1

        logger.info(
            f"Ingested Polymarket: {new_count} new snapshots "
            f"({len(markets) - new_count} duplicates skipped)"
        )
        return new_count


# Convenience function for quick testing
def fetch_sample_markets(limit: int = 5) -> list[dict]:
    """
    Fetch a few sample geopolitical prediction markets for testing.

    Args:
        limit: Number of events to fetch (markets may be more)

    Returns:
        List of market dicts
    """
    pm = PolymarketIngestion()
    markets = pm.fetch_markets(limit=limit)
    return markets[:20]  # Cap output for readability
