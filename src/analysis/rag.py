"""
RAG (Retrieval-Augmented Generation) over the event database.

Enables natural language queries against the geopolitical event database.
Users can ask questions like "What military conflicts involved NATO this month?"
and get sourced answers from the data.

HOW RAG WORKS:
--------------
1. EMBED: Convert all events into vector embeddings
2. INDEX: Store embeddings in a searchable index (FAISS)
3. QUERY: When a user asks a question, embed the question
4. RETRIEVE: Find the most similar events by vector similarity
5. ANSWER: Format retrieved events into a structured response

This is the same pattern used by Palantir AIP and other defense
intelligence platforms.

USAGE:
------
    from src.analysis.rag import EventRAG

    rag = EventRAG()
    rag.build_index(start_date, end_date)
    answer = rag.query("What sanctions were imposed this month?")
    print(answer.response)
    print(answer.sources)
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional
import logging

import numpy as np
import pandas as pd

from src.db.connection import get_session
from src.db.queries import get_events_by_date_range

logger = logging.getLogger(__name__)


@dataclass
class RetrievedEvent:
    """A single retrieved event with relevance score."""
    event_id: int
    event_date: date
    text: str
    relevance_score: float
    actor1: str = ""
    actor2: str = ""
    goldstein_scale: float = 0.0
    num_mentions: int = 0
    country: str = ""


@dataclass
class RAGResponse:
    """Response from a RAG query."""
    query: str
    response: str
    sources: list[RetrievedEvent]
    total_events_searched: int
    retrieval_method: str = "semantic_similarity"


class EventRAG:
    """
    Retrieval-Augmented Generation over geopolitical events.

    Uses sentence-transformers for embedding and FAISS (or numpy
    fallback) for similarity search.
    """

    def __init__(self):
        self._embedder = None
        self._index = None
        self._events: list[dict] = []
        self._event_texts: list[str] = []
        self._embeddings: Optional[np.ndarray] = None

    def _get_embedder(self):
        """Lazy-load the event embedder."""
        if self._embedder is not None:
            return self._embedder

        try:
            from src.analysis.text_embeddings import EventEmbedder

            self._embedder = EventEmbedder()
            return self._embedder
        except ImportError:
            logger.error(
                "sentence-transformers not available. "
                "Install: pip install sentence-transformers"
            )
            return None

    def build_index(
        self,
        start_date: date,
        end_date: date,
        limit: int = 1000,
    ) -> int:
        """
        Build the search index from events in the database.

        Fetches events, converts to text, embeds, and indexes.
        Returns the number of events indexed.
        """
        embedder = self._get_embedder()
        if embedder is None:
            return 0

        # Fetch events
        with get_session() as session:
            events_raw = get_events_by_date_range(
                session, start_date, end_date, limit=limit
            )

        if not events_raw:
            logger.warning("No events found for indexing")
            return 0

        # Convert to dicts and generate text
        self._events = []
        self._event_texts = []

        for e in events_raw:
            event_dict = {
                "id": e.id,
                "event_date": e.event_date,
                "event_root_code": e.event_root_code,
                "actor1_name": e.actor1_name,
                "actor1_code": e.actor1_code,
                "actor2_name": e.actor2_name,
                "actor2_code": e.actor2_code,
                "action_geo_name": getattr(e, "action_geo_name", None),
                "action_geo_country_code": getattr(e, "action_geo_country_code", None),
                "goldstein_scale": e.goldstein_scale,
                "avg_tone": e.avg_tone,
                "num_mentions": e.num_mentions,
            }
            self._events.append(event_dict)
            self._event_texts.append(embedder.event_to_text(event_dict))

        # Embed all texts
        embeddings_list = embedder.embed_batch(self._event_texts, batch_size=64)
        self._embeddings = np.array(embeddings_list)

        # Build FAISS index if available, otherwise use numpy
        try:
            import faiss

            dimension = self._embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dimension)  # Inner product (cosine sim)

            # Normalize for cosine similarity
            faiss.normalize_L2(self._embeddings)
            self._index.add(self._embeddings)

            logger.info(
                f"Built FAISS index with {len(self._events)} events"
            )
        except ImportError:
            # Numpy fallback - normalize for cosine similarity
            norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
            self._embeddings = self._embeddings / (norms + 1e-10)
            self._index = None  # Will use numpy dot product
            logger.info(
                f"Built numpy index with {len(self._events)} events (FAISS not available)"
            )

        return len(self._events)

    def _search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Search the index for similar events.

        Returns list of (index, score) tuples.
        """
        if self._embeddings is None or len(self._events) == 0:
            return []

        query_vec = query_embedding.reshape(1, -1).astype("float32")

        try:
            import faiss

            if self._index is not None:
                faiss.normalize_L2(query_vec)
                scores, indices = self._index.search(query_vec, min(top_k, len(self._events)))
                return [
                    (int(idx), float(score))
                    for idx, score in zip(indices[0], scores[0])
                    if idx >= 0
                ]
        except ImportError:
            pass

        # Numpy fallback
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        similarities = np.dot(self._embeddings, query_norm.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            (int(idx), float(similarities[idx]))
            for idx in top_indices
        ]

    def query(
        self,
        question: str,
        top_k: int = 10,
        min_relevance: float = 0.3,
    ) -> RAGResponse:
        """
        Query the event database with natural language.

        Args:
            question: Natural language query
            top_k: Maximum number of events to retrieve
            min_relevance: Minimum relevance score (0-1)

        Returns:
            RAGResponse with formatted answer and source events
        """
        embedder = self._get_embedder()
        if embedder is None:
            return RAGResponse(
                query=question,
                response="Embedding model not available.",
                sources=[],
                total_events_searched=0,
            )

        if self._embeddings is None or len(self._events) == 0:
            return RAGResponse(
                query=question,
                response="No events indexed. Call build_index() first.",
                sources=[],
                total_events_searched=0,
            )

        # Embed the query
        query_embedding = embedder.embed_text(question)

        # Search
        results = self._search(query_embedding, top_k=top_k)

        # Filter by relevance and build response
        sources = []
        for idx, score in results:
            if score < min_relevance:
                continue

            event = self._events[idx]
            sources.append(
                RetrievedEvent(
                    event_id=event["id"],
                    event_date=event["event_date"],
                    text=self._event_texts[idx],
                    relevance_score=score,
                    actor1=event.get("actor1_name") or event.get("actor1_code") or "",
                    actor2=event.get("actor2_name") or event.get("actor2_code") or "",
                    goldstein_scale=event.get("goldstein_scale") or 0.0,
                    num_mentions=event.get("num_mentions") or 0,
                    country=event.get("action_geo_country_code") or "",
                )
            )

        # Format response
        response = self._format_response(question, sources)

        return RAGResponse(
            query=question,
            response=response,
            sources=sources,
            total_events_searched=len(self._events),
        )

    def _format_response(
        self,
        question: str,
        sources: list[RetrievedEvent],
    ) -> str:
        """Format retrieved events into a structured answer."""
        if not sources:
            return f"No relevant events found for: '{question}'"

        lines = [
            f"Found {len(sources)} relevant events for: '{question}'",
            "",
        ]

        for i, source in enumerate(sources, 1):
            severity = ""
            if source.goldstein_scale <= -5:
                severity = " [HIGH SEVERITY]"
            elif source.goldstein_scale <= -2:
                severity = " [MODERATE]"

            lines.append(
                f"{i}. [{source.event_date}] {source.text}{severity}"
            )
            lines.append(
                f"   Relevance: {source.relevance_score:.0%} | "
                f"Goldstein: {source.goldstein_scale:.1f} | "
                f"Mentions: {source.num_mentions}"
            )
            lines.append("")

        # Summary statistics
        avg_goldstein = sum(s.goldstein_scale for s in sources) / len(sources)
        total_mentions = sum(s.num_mentions for s in sources)

        lines.extend([
            "---",
            f"Average Goldstein: {avg_goldstein:.1f}",
            f"Total media mentions: {total_mentions}",
            f"Date range: {sources[-1].event_date} to {sources[0].event_date}",
        ])

        return "\n".join(lines)

    def find_similar_events(
        self,
        event_dict: dict,
        top_k: int = 5,
    ) -> list[RetrievedEvent]:
        """
        Find events similar to a given event.

        Useful for finding historical parallels.
        """
        embedder = self._get_embedder()
        if embedder is None or self._embeddings is None:
            return []

        text = embedder.event_to_text(event_dict)
        embedding = embedder.embed_text(text)

        results = self._search(embedding, top_k=top_k + 1)

        # Skip the event itself if it's in the index
        sources = []
        for idx, score in results:
            event = self._events[idx]
            if event.get("id") == event_dict.get("id"):
                continue

            sources.append(
                RetrievedEvent(
                    event_id=event["id"],
                    event_date=event["event_date"],
                    text=self._event_texts[idx],
                    relevance_score=score,
                    actor1=event.get("actor1_name") or event.get("actor1_code") or "",
                    actor2=event.get("actor2_name") or event.get("actor2_code") or "",
                    goldstein_scale=event.get("goldstein_scale") or 0.0,
                    num_mentions=event.get("num_mentions") or 0,
                    country=event.get("action_geo_country_code") or "",
                )
            )

            if len(sources) >= top_k:
                break

        return sources
