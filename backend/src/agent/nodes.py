"""LangGraph agent nodes — deterministic collection/analysis, LLM synthesis.

Each node is a function that:
  1. Receives the shared AgentState
  2. Performs its work (data fetching, analysis, or LLM synthesis)
  3. Returns a state update dict (LangGraph merges it into state)

Architecture:
  Collection Agent    -> deterministic data gathering -> writes state.collected_data
  Analysis Agent      -> deterministic analysis       -> writes state.analysis_results
  Dissemination Agent -> Ollama LLM synthesis         -> writes state.final_response
"""

import json
import logging
from dataclasses import asdict
from datetime import date, timedelta

from src.agent.state import AgentState, CollectedData, AnalysisResults
from src.agent.tools import execute_tool
from src.analysis.synthesis import generate_assessment
from src.config.constants import COUNTRY_ASSET_MAP, SYMBOL_COUNTRY_MAP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: extract country codes from a query string
# ---------------------------------------------------------------------------

# Common country name -> code mappings for query parsing
_COUNTRY_KEYWORDS = {
    "russia": "RUS", "russian": "RUS", "moscow": "RUS", "kremlin": "RUS",
    "china": "CHN", "chinese": "CHN", "beijing": "CHN",
    "iran": "IRN", "iranian": "IRN", "tehran": "IRN",
    "ukraine": "UKR", "ukrainian": "UKR", "kyiv": "UKR",
    "israel": "ISR", "israeli": "ISR",
    "saudi": "SAU", "saudi arabia": "SAU", "riyadh": "SAU",
    "brazil": "BRA", "brazilian": "BRA",
    "japan": "JPN", "japanese": "JPN",
    "germany": "DEU", "german": "DEU",
    "taiwan": "TWN", "taiwanese": "TWN",
    "korea": "KOR", "korean": "KOR",
    "india": "IND", "indian": "IND",
    "uk": "GBR", "britain": "GBR", "british": "GBR",
    "venezuela": "VEN",
    "nigeria": "NGA",
}


def _extract_countries(query: str) -> list[str]:
    """Extract country codes from a natural language query."""
    query_lower = query.lower()
    found = set()
    for keyword, code in _COUNTRY_KEYWORDS.items():
        if keyword in query_lower:
            found.add(code)
    return list(found)


def _get_symbols_for_countries(countries: list[str]) -> list[str]:
    """Get relevant financial symbols for a list of country codes."""
    symbols = set()
    for country in countries:
        if country in COUNTRY_ASSET_MAP:
            symbols.update(COUNTRY_ASSET_MAP[country])
    # Always include broad market indicators
    symbols.update(["SPY", "GC=F", "CL=F", "^VIX"])
    return list(symbols)


# ---------------------------------------------------------------------------
# Node functions — these are what LangGraph calls
# ---------------------------------------------------------------------------

def collection_node(state: AgentState) -> dict:
    """Collection Agent: gather raw data relevant to the query.

    Deterministic — uses query parsing and country/asset mappings
    to decide what data to fetch. No LLM needed.
    """
    query = state["query"]
    logger.info(f"[Collection] Processing: {query}")

    collected = CollectedData()
    end_date = date.today()
    start_30d = end_date - timedelta(days=30)
    start_7d = end_date - timedelta(days=7)

    # 1. Extract countries from query and fetch relevant events
    countries = _extract_countries(query)

    if countries:
        for country in countries:
            result = execute_tool("get_recent_events", {
                "country_code": country,
                "start_date": str(start_30d),
                "end_date": str(end_date),
                "limit": 15,
            })
            collected.events.append({"country": country, "data": json.loads(result)})
    else:
        # No specific country — get global event summary
        result = execute_tool("get_event_summary", {
            "start_date": str(start_30d),
            "end_date": str(end_date),
            "group_by": "country",
        })
        collected.events.append({"global_summary": json.loads(result)})

    # 2. Fetch market data for relevant symbols
    symbols = _get_symbols_for_countries(countries) if countries else [
        "SPY", "CL=F", "GC=F", "^VIX", "TLT", "EEM",
    ]

    for symbol in symbols[:8]:  # Cap at 8 to keep context manageable
        result = execute_tool("get_market_data", {
            "symbol": symbol,
            "start_date": str(start_7d),
            "end_date": str(end_date),
            "limit": 7,
        })
        collected.market_data.append({"symbol": symbol, "data": json.loads(result)})

    # 3. Fetch headline sentiment
    sentiment_params = {
        "start_date": str(start_7d),
        "end_date": str(end_date),
        "limit": 15,
    }
    result = execute_tool("get_headline_sentiment", sentiment_params)
    collected.headlines.append(json.loads(result))

    # 4. RAG search for broader context
    result = execute_tool("rag_search", {"query": query, "days_back": 30, "limit": 10})
    rag_data = json.loads(result)
    collected.rag_context = rag_data.get("context", "")

    logger.info(
        f"[Collection] Done: {len(collected.events)} event queries, "
        f"{len(collected.market_data)} symbols, "
        f"{len(collected.headlines)} headline sets"
    )

    return {
        "collected_data": collected,
        "current_agent": "collection",
        "messages": [{"role": "assistant", "content": "[Collection] Data gathering complete."}],
    }


def analysis_node(state: AgentState) -> dict:
    """Analysis Agent: produce insights from collected data.

    Deterministic — runs all relevant analytical tools on the
    collected data. No LLM needed for analysis decisions.
    """
    query = state["query"]
    logger.info(f"[Analysis] Processing: {query}")

    collected = state.get("collected_data", CollectedData())
    results = AnalysisResults()

    # Extract symbols from collected market data
    symbols = [
        md["symbol"] for md in collected.market_data
        if isinstance(md, dict) and "symbol" in md
    ]

    # 1. Get sentiment summary
    result = execute_tool("get_sentiment_summary", {
        "start_date": str(date.today() - timedelta(days=7)),
        "end_date": str(date.today()),
    })
    results.sentiment_summary = json.loads(result)

    # 2. Run analysis for each collected symbol
    for symbol in symbols[:6]:  # Cap to keep reasonable
        # Correlations
        corr_result = execute_tool("get_correlations", {"symbol": symbol})
        corr_data = json.loads(corr_result)
        significant = [c for c in corr_data if isinstance(c, dict) and c.get("significant")]
        if significant:
            results.key_findings.append(
                f"{symbol}: {len(significant)} significant correlations found"
            )

        # Historical patterns
        pattern_result = execute_tool("get_historical_patterns", {"symbol": symbol})
        pattern_data = json.loads(pattern_result)
        sig_patterns = [p for p in pattern_data if isinstance(p, dict) and p.get("is_significant")]
        if sig_patterns:
            results.patterns.append({"symbol": symbol, "patterns": sig_patterns[:3]})

        # Anomaly detection (recent 90 days)
        anomaly_result = execute_tool("detect_anomalies", {"symbol": symbol})
        anomaly_data = json.loads(anomaly_result)
        if isinstance(anomaly_data, dict) and anomaly_data.get("total_anomalies", 0) > 0:
            results.anomalies.append(anomaly_data)

    # 3. Get top correlations across all symbols
    top_corr = execute_tool("get_top_correlations", {"limit": 10})
    top_corr_data = json.loads(top_corr)
    if top_corr_data:
        results.key_findings.append(
            f"Top cross-market correlations: {len(top_corr_data)} pairs identified"
        )

    logger.info(
        f"[Analysis] Done: {len(results.key_findings)} findings, "
        f"{len(results.patterns)} pattern sets, "
        f"{len(results.anomalies)} anomaly reports"
    )

    return {
        "analysis_results": results,
        "current_agent": "analysis",
        "messages": [{"role": "assistant", "content": "[Analysis] Analysis complete."}],
    }


def dissemination_node(state: AgentState) -> dict:
    """Dissemination Agent: synthesize everything into a final briefing.

    This is the ONLY node that uses an LLM (Ollama). It takes the
    structured data from collection and analysis and produces a
    readable intelligence assessment.
    """
    query = state["query"]
    logger.info(f"[Dissemination] Processing: {query}")

    collected = state.get("collected_data", CollectedData())
    analysis = state.get("analysis_results", AnalysisResults())

    # Build structured input for the LLM
    structured_data = (
        f"USER QUERY: {query}\n\n"
        f"=== COLLECTED DATA ===\n"
        f"{json.dumps(asdict(collected), default=str, indent=2)}\n\n"
        f"=== ANALYSIS RESULTS ===\n"
        f"{json.dumps(asdict(analysis), default=str, indent=2)}"
    )

    # Generate assessment via local LLM
    response_text = generate_assessment(structured_data)

    return {
        "final_response": response_text,
        "current_agent": "dissemination",
        "messages": [{"role": "assistant", "content": response_text}],
    }
