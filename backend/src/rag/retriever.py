"""
RAG Retriever — vector similarity search over headlines and events.

HOW IT WORKS:
-------------
1. User asks a question (e.g., "What's happening with Iran sanctions?")
2. We embed the question into a 384-dim vector using the same MiniLM model
3. We query pgvector for the closest vectors using cosine distance (<=>)
4. Return the top-K most semantically similar documents

WHY THIS PATTERN:
- The retriever is SEPARATE from the context builder (Step 20) and the
  agent tool (Step 21). Each layer has one job:
  - Retriever: "find relevant documents" (this file)
  - Context builder: "format documents into a prompt" (context.py)
  - Agent tool: "expose RAG to Claude" (tools.py)

DISTANCE THRESHOLDS:
- 0.0-0.3: Very similar (almost paraphrasing)
- 0.3-0.6: Related topic
- 0.6-0.8: Loosely related
- 0.8+: Probably not relevant — we filter these out

USAGE:
    retriever = Retriever()
    results = retriever.search_headlines("nuclear escalation threats", limit=10)
    # Returns list of RetrievedDoc with text, metadata, and distance score
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

from src.analysis.embeddings import EmbeddingGenerator
from src.db.connection import get_session

logger = logging.getLogger(__name__)

# Documents beyond this distance are filtered out as irrelevant
# Cosine distance range is 0 (identical) to 2 (opposite). 1.0 is orthogonal.
# 0.9 keeps loosely-related content, filters true noise.
MAX_DISTANCE = 0.9


@dataclass
class RetrievedDoc:
    """A single document returned by the retriever."""
    content: str                        # The text content
    source_type: str                    # "headline" or "event"
    distance: float                     # Cosine distance (0 = identical, 2 = opposite)
    metadata: dict = field(default_factory=dict)  # Extra fields (date, source, sentiment, etc.)


class Retriever:
    """
    Searches headlines and events by semantic similarity.

    Uses the same MiniLM embedding model that was used to embed the content,
    ensuring the query vector lives in the same vector space as the stored vectors.
    """

    def __init__(self, max_distance: float = MAX_DISTANCE):
        self.max_distance = max_distance
        self._generator = None

    def _get_generator(self) -> EmbeddingGenerator:
        """Lazy-load the embedding generator."""
        if self._generator is None:
            self._generator = EmbeddingGenerator()
        return self._generator

    def _embed_query(self, query: str) -> list[float]:
        """Convert a query string to a 384-dim vector."""
        generator = self._get_generator()
        vectors = generator.encode([query])
        return vectors[0].tolist()

    def search_headlines(
        self,
        query: str,
        limit: int = 10,
        start_date: date | None = None,
        end_date: date | None = None,
        source: str | None = None,
    ) -> list[RetrievedDoc]:
        """
        Find headlines most similar to the query.

        Args:
            query: Natural language search query
            limit: Max results to return
            start_date: Optional date filter (inclusive)
            end_date: Optional date filter (inclusive)
            source: Optional source filter ("reuters", "ap", "bbc", "aljazeera")

        Returns:
            List of RetrievedDoc, sorted by relevance (closest first)
        """
        from src.db.models import NewsHeadline
        from sqlalchemy import func

        query_embedding = self._embed_query(query)

        with get_session() as session:
            q = session.query(
                NewsHeadline,
                NewsHeadline.embedding.cosine_distance(query_embedding).label("distance"),
            ).filter(
                NewsHeadline.embedding.isnot(None),
            )

            if start_date:
                q = q.filter(func.date(NewsHeadline.published_at) >= start_date)
            if end_date:
                q = q.filter(func.date(NewsHeadline.published_at) <= end_date)
            if source:
                q = q.filter(NewsHeadline.source == source)

            rows = q.order_by(
                NewsHeadline.embedding.cosine_distance(query_embedding),
            ).limit(limit).all()

        results = []
        for row in rows:
            headline = row.NewsHeadline
            dist = float(row.distance)

            if dist > self.max_distance:
                continue

            results.append(RetrievedDoc(
                content=headline.headline,
                source_type="headline",
                distance=dist,
                metadata={
                    "id": headline.id,
                    "source": headline.source,
                    "published_at": str(headline.published_at),
                    "sentiment_score": headline.sentiment_score,
                    "sentiment_label": headline.sentiment_label,
                    "url": headline.url,
                },
            ))

        return results

    def search_events(
        self,
        query: str,
        limit: int = 10,
        start_date: date | None = None,
        end_date: date | None = None,
        country_code: str | None = None,
    ) -> list[RetrievedDoc]:
        """
        Find events most similar to the query.

        Args:
            query: Natural language search query
            limit: Max results to return
            start_date: Optional date filter
            end_date: Optional date filter
            country_code: Optional 3-letter ISO country code filter

        Returns:
            List of RetrievedDoc, sorted by relevance (closest first)
        """
        from src.db.models import Event

        query_embedding = self._embed_query(query)

        with get_session() as session:
            q = session.query(
                Event,
                Event.embedding.cosine_distance(query_embedding).label("distance"),
            ).filter(
                Event.embedding.isnot(None),
            )

            if start_date:
                q = q.filter(Event.event_date >= start_date)
            if end_date:
                q = q.filter(Event.event_date <= end_date)
            if country_code:
                q = q.filter(Event.action_geo_country_code == country_code.upper())

            rows = q.order_by(
                Event.embedding.cosine_distance(query_embedding),
            ).limit(limit).all()

        results = []
        for row in rows:
            event = row.Event
            dist = float(row.distance)

            if dist > self.max_distance:
                continue

            # Build a readable summary for the context
            actor1 = event.actor1_name or event.actor1_code or "Unknown"
            actor2 = event.actor2_name or event.actor2_code or "Unknown"
            location = event.action_geo_name or event.action_geo_country_code or "Unknown"
            content = f"{actor1} → {actor2} in {location}"

            results.append(RetrievedDoc(
                content=content,
                source_type="event",
                distance=dist,
                metadata={
                    "id": event.id,
                    "event_date": str(event.event_date),
                    "goldstein_scale": event.goldstein_scale,
                    "num_mentions": event.num_mentions,
                    "event_root_code": event.event_root_code,
                    "actor1_country": event.actor1_country_code,
                    "actor2_country": event.actor2_country_code,
                    "location": location,
                },
            ))

        return results

    def search_all(
        self,
        query: str,
        limit: int = 15,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[RetrievedDoc]:
        """
        Search both headlines and events, merge by relevance.

        Splits the limit between headlines and events, then
        interleaves results sorted by distance.
        """
        half = max(limit // 2, 5)

        headlines = self.search_headlines(
            query, limit=half, start_date=start_date, end_date=end_date,
        )
        events = self.search_events(
            query, limit=half, start_date=start_date, end_date=end_date,
        )

        # Merge and sort by distance (most relevant first)
        combined = headlines + events
        combined.sort(key=lambda d: d.distance)

        return combined[:limit]
