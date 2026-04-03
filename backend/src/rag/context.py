"""
RAG Context Builder — assembles retrieved documents into LLM prompt context.

HOW IT WORKS:
-------------
1. Retriever finds the top-K most relevant headlines and events
2. Context builder formats them into a structured text block
3. This block gets injected into the LLM prompt as grounding context
4. The LLM generates a response based on REAL data, not hallucination

WHY A SEPARATE MODULE:
- Retriever knows HOW to search (vector math, SQL queries)
- Context builder knows HOW TO FORMAT for the LLM (prompt engineering)
- Keeping them separate means we can change formatting without touching search logic

CONTEXT FORMAT:
    The context is structured as a markdown-like block that the LLM can parse:

    === RELEVANT INTELLIGENCE ===

    HEADLINES (5 results):
    [2026-03-09] (reuters, sentiment: -0.82) Oil prices surge as Iran tensions escalate
    [2026-03-08] (ap, sentiment: +0.45) EU announces new diplomatic initiative
    ...

    EVENTS (5 results):
    [2026-03-09] USA → IRN in Tehran (Goldstein: -7.0, mentions: 450)
    ...

USAGE:
    retriever = Retriever()
    builder = ContextBuilder(retriever)
    context = builder.build("What's happening with Iran?")
    # Returns formatted string ready to inject into a prompt
"""

import logging
from datetime import date, timedelta

from src.rag.retriever import Retriever, RetrievedDoc

logger = logging.getLogger(__name__)

# Max tokens (approximate) for the context block to avoid bloating the prompt
MAX_CONTEXT_CHARS = 4000


class ContextBuilder:
    """
    Builds structured context from retrieved documents for LLM consumption.

    The builder calls the retriever, formats results, and returns a
    string that can be injected directly into a system or user prompt.
    """

    def __init__(self, retriever: Retriever | None = None):
        self.retriever = retriever or Retriever()

    def build(
        self,
        query: str,
        limit: int = 15,
        start_date: date | None = None,
        end_date: date | None = None,
        include_headlines: bool = True,
        include_events: bool = True,
    ) -> str:
        """
        Build a context string for the given query.

        Args:
            query: The user's question or topic
            limit: Max total documents to retrieve
            start_date: Optional date filter
            end_date: Optional date filter
            include_headlines: Whether to search headlines
            include_events: Whether to search events

        Returns:
            Formatted context string, or empty string if nothing relevant found
        """
        docs = []

        if include_headlines and include_events:
            docs = self.retriever.search_all(
                query, limit=limit, start_date=start_date, end_date=end_date,
            )
        elif include_headlines:
            docs = self.retriever.search_headlines(
                query, limit=limit, start_date=start_date, end_date=end_date,
            )
        elif include_events:
            docs = self.retriever.search_events(
                query, limit=limit, start_date=start_date, end_date=end_date,
            )

        if not docs:
            return ""

        return self._format_docs(docs, query)

    def build_briefing_context(
        self,
        topics: list[str] | None = None,
        days_back: int = 3,
        limit_per_topic: int = 5,
    ) -> str:
        """
        Build context for the Intelligence Briefing AI summary.

        Instead of a single query, this searches multiple topics
        to provide broad situational awareness.

        Args:
            topics: List of topics to search. Defaults to general geopolitical topics.
            days_back: How far back to search
            limit_per_topic: Results per topic

        Returns:
            Formatted context covering multiple topics
        """
        if topics is None:
            topics = [
                "military conflict and armed confrontation",
                "diplomatic negotiations and peace talks",
                "economic sanctions and trade disputes",
                "energy markets oil gas supply disruption",
                "political instability elections regime change",
            ]

        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)

        all_docs: list[RetrievedDoc] = []
        seen_content: set[str] = set()

        for topic in topics:
            docs = self.retriever.search_all(
                topic,
                limit=limit_per_topic,
                start_date=start_date,
                end_date=end_date,
            )
            for doc in docs:
                # Deduplicate across topics
                if doc.content not in seen_content:
                    seen_content.add(doc.content)
                    all_docs.append(doc)

        if not all_docs:
            return ""

        # Sort by distance (most relevant across all topics)
        all_docs.sort(key=lambda d: d.distance)

        return self._format_docs(all_docs, "situational awareness briefing")

    def _format_docs(self, docs: list[RetrievedDoc], query: str) -> str:
        """Format retrieved documents into a structured context block."""
        headlines = [d for d in docs if d.source_type == "headline"]
        events = [d for d in docs if d.source_type == "event"]

        parts = [f"=== RELEVANT INTELLIGENCE (query: {query}) ===\n"]

        if headlines:
            parts.append(f"HEADLINES ({len(headlines)} results):")
            for h in headlines:
                date_str = h.metadata.get("published_at", "unknown")[:10]
                source = h.metadata.get("source", "unknown")
                sentiment = h.metadata.get("sentiment_score")
                sent_str = f", sentiment: {sentiment:+.2f}" if sentiment is not None else ""
                parts.append(f"  [{date_str}] ({source}{sent_str}) {h.content}")
            parts.append("")

        if events:
            parts.append(f"EVENTS ({len(events)} results):")
            for e in events:
                date_str = e.metadata.get("event_date", "unknown")
                goldstein = e.metadata.get("goldstein_scale")
                mentions = e.metadata.get("num_mentions")
                g_str = f", Goldstein: {goldstein}" if goldstein is not None else ""
                m_str = f", mentions: {mentions}" if mentions else ""
                parts.append(f"  [{date_str}] {e.content} ({g_str}{m_str})")
            parts.append("")

        context = "\n".join(parts)

        # Truncate if too long
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS] + "\n... (truncated)"

        return context
