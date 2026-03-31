"""
MCP Server for the Geopolitical Market Tracker.

Registers all tool functions from the agent tools module as MCP tools,
making them available to Claude Desktop and Claude Code via stdio transport.
"""

import json
import logging

from mcp.server.fastmcp import FastMCP

from src.agent.tools import (
    _exec_get_recent_events,
    _exec_get_event_summary,
    _exec_get_market_data,
    _exec_get_correlations,
    _exec_get_top_correlations,
    _exec_get_historical_patterns,
    _exec_run_prediction,
    _exec_detect_anomalies,
    _exec_list_symbols,
    _exec_get_symbol_countries,
    _exec_get_headline_sentiment,
    _exec_get_sentiment_summary,
    _exec_search_similar_content,
    _exec_rag_search,
)
from src.rag.context import ContextBuilder

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "Geopolitical Market Tracker",
    instructions=(
        "Intelligence platform fusing geopolitical events, financial markets, "
        "economic indicators, news sentiment, and prediction market data. "
        "Provides correlation analysis, predictions, anomaly detection, and RAG search."
    ),
)


# =============================================================================
# Event Tools
# =============================================================================

@mcp.tool()
def get_recent_events(
    start_date: str | None = None,
    end_date: str | None = None,
    country_code: str | None = None,
    min_mentions: int = 200,
    limit: int = 20,
) -> str:
    """Query GDELT geopolitical events with filters. Returns events with date, actors,
    Goldstein scale, mentions, and location. Use this to find what happened in a
    country or time period."""
    result = _exec_get_recent_events({
        "start_date": start_date,
        "end_date": end_date,
        "country_code": country_code,
        "min_mentions": min_mentions,
        "limit": limit,
    })
    return json.dumps(result, default=str)


@mcp.tool()
def get_event_summary(
    start_date: str | None = None,
    end_date: str | None = None,
    group_by: str = "country",
) -> str:
    """Get event counts by country or by event type for a date range.
    Use this to understand event distribution and hotspots."""
    result = _exec_get_event_summary({
        "start_date": start_date,
        "end_date": end_date,
        "group_by": group_by,
    })
    return json.dumps(result, default=str)


# =============================================================================
# Market Tools
# =============================================================================

@mcp.tool()
def get_market_data(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 30,
) -> str:
    """Get daily OHLCV market data and returns for a financial instrument.
    Symbols include commodities (CL=F, GC=F), currencies (EURUSD=X),
    ETFs (SPY, EEM), and more."""
    result = _exec_get_market_data({
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "limit": limit,
    })
    return json.dumps(result, default=str)


@mcp.tool()
def list_symbols() -> str:
    """List all 33 tracked financial instruments organized by category:
    commodities, currencies, ETFs, volatility, bonds."""
    result = _exec_list_symbols({})
    return json.dumps(result, default=str)


@mcp.tool()
def get_symbol_countries(symbol: str | None = None) -> str:
    """Get the country-to-symbol sensitivity mappings. Shows which countries
    affect which financial instruments (e.g., Russia -> Ruble, Oil, Natural Gas)."""
    result = _exec_get_symbol_countries({"symbol": symbol} if symbol else {})
    return json.dumps(result, default=str)


# =============================================================================
# Analysis Tools
# =============================================================================

@mcp.tool()
def get_correlations(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """Compute Pearson correlation between geopolitical event metrics and daily
    log returns for a symbol. Shows how events relate to market moves."""
    result = _exec_get_correlations({
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
    })
    return json.dumps(result, default=str)


@mcp.tool()
def get_top_correlations(limit: int = 20) -> str:
    """Get the strongest event-market correlations across all 33 tracked symbols.
    Uses precomputed cache for fast results."""
    result = _exec_get_top_correlations({"limit": limit})
    return json.dumps(result, default=str)


@mcp.tool()
def get_historical_patterns(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """Analyze historical frequency patterns: 'When X event happens, Y symbol
    goes UP Z% of the time.' Checks all event groups and country-specific patterns."""
    result = _exec_get_historical_patterns({
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
    })
    return json.dumps(result, default=str)


@mcp.tool()
def run_prediction(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """Run a logistic regression prediction for a symbol's next-day market direction
    (UP/DOWN). Uses 7 event-based features trained on historical data. Returns
    probability, accuracy, and key feature contributions."""
    result = _exec_run_prediction({
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
    })
    return json.dumps(result, default=str)


@mcp.tool()
def detect_anomalies(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """Detect unusual market-event mismatches using Isolation Forest, Z-scores,
    and domain rules. Finds unexplained moves, muted responses, and statistical outliers."""
    result = _exec_detect_anomalies({
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
    })
    return json.dumps(result, default=str)


# =============================================================================
# Sentiment Tools
# =============================================================================

@mcp.tool()
def get_headline_sentiment(
    start_date: str | None = None,
    end_date: str | None = None,
    source: str | None = None,
    sentiment: str | None = None,
    limit: int = 20,
) -> str:
    """Get recent news headlines with their FinBERT sentiment scores.
    Each headline has a score (-1.0 negative to +1.0 positive) and label.
    Sources: reuters, ap, bbc, aljazeera."""
    result = _exec_get_headline_sentiment({
        "start_date": start_date,
        "end_date": end_date,
        "source": source,
        "sentiment": sentiment,
        "limit": limit,
    })
    return json.dumps(result, default=str)


@mcp.tool()
def get_sentiment_summary(
    start_date: str | None = None,
    end_date: str | None = None,
    source: str | None = None,
) -> str:
    """Get aggregate sentiment statistics for news headlines over a time period.
    Returns average score, distribution, and daily trend."""
    result = _exec_get_sentiment_summary({
        "start_date": start_date,
        "end_date": end_date,
        "source": source,
    })
    return json.dumps(result, default=str)


# =============================================================================
# Search & RAG Tools
# =============================================================================

@mcp.tool()
def search_similar_content(
    query: str,
    content_type: str = "headlines",
    limit: int = 10,
) -> str:
    """Semantic search using embeddings. Given a natural language query, finds
    the most similar headlines or events by meaning (not just keyword match).
    content_type: 'headlines' or 'events'."""
    result = _exec_search_similar_content({
        "query": query,
        "content_type": content_type,
        "limit": limit,
    })
    return json.dumps(result, default=str)


@mcp.tool()
def rag_search(
    query: str,
    days_back: int = 30,
    limit: int = 15,
) -> str:
    """RAG search: retrieves the most relevant headlines AND events using semantic
    similarity and returns them as structured intelligence context. Use for broad
    research questions or comprehensive background on a topic."""
    result = _exec_rag_search({
        "query": query,
        "days_back": days_back,
        "limit": limit,
    })
    return json.dumps(result, default=str)


@mcp.tool()
def get_briefing_context(
    topics: list[str] | None = None,
    days_back: int = 3,
    limit_per_topic: int = 5,
) -> str:
    """Build a multi-topic situational awareness briefing. Searches across military
    conflict, diplomacy, sanctions, energy markets, and political instability.
    Returns structured context covering multiple geopolitical themes."""
    builder = ContextBuilder()
    context = builder.build_briefing_context(
        topics=topics,
        days_back=days_back,
        limit_per_topic=limit_per_topic,
    )
    if not context:
        return json.dumps({"message": "No relevant content found for briefing."})
    return context
