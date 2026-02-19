"""Agent tool definitions and execution dispatch.

Each tool maps a Claude tool_use schema to an internal Python function.
The agent calls these tools directly (no HTTP overhead).
"""

import json
import logging
from dataclasses import asdict
from datetime import date, timedelta

from src.config.constants import (
    SYMBOLS,
    get_all_symbols,
    get_symbol_info,
    SYMBOL_COUNTRY_MAP,
    COUNTRY_ASSET_MAP,
    EVENT_GROUPS,
    CAMEO_CATEGORIES,
)
from src.db.connection import get_session
from src.db.queries import (
    get_events_by_date_range,
    get_event_counts_by_country,
    get_event_counts_by_type,
    get_market_data,
    get_cached_correlations,
)
from src.analysis.correlation import CorrelationAnalyzer
from src.analysis.historical_patterns import HistoricalPatternAnalyzer
from src.analysis.production_regression import LogisticRegressionAnalyzer
from src.analysis.production_anomaly import ProductionAnomalyDetector

logger = logging.getLogger(__name__)

# =============================================================================
# TOOL SCHEMAS (Claude tool_use format)
# =============================================================================

TOOL_DEFINITIONS = [
    {
        "name": "get_recent_events",
        "description": (
            "Query GDELT geopolitical events with filters. Returns events with "
            "date, actors, Goldstein scale, mentions, and location. Use this to "
            "find what happened in a country or time period."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "Start date (YYYY-MM-DD). Defaults to 30 days ago.",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date (YYYY-MM-DD). Defaults to today.",
                },
                "country_code": {
                    "type": "string",
                    "description": "3-letter ISO country code (e.g. RUS, CHN, USA, IRN, UKR).",
                },
                "min_mentions": {
                    "type": "integer",
                    "description": "Minimum media mentions. Higher = more significant. Default 200.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max events to return. Default 20.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_event_summary",
        "description": (
            "Get event counts by country or by event type for a date range. "
            "Use this to understand event distribution and hotspots."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "Start date (YYYY-MM-DD). Defaults to 30 days ago.",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date (YYYY-MM-DD). Defaults to today.",
                },
                "group_by": {
                    "type": "string",
                    "enum": ["country", "type"],
                    "description": "Group counts by country or event type. Default: country.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_market_data",
        "description": (
            "Get daily OHLCV market data and returns for a financial instrument. "
            "Symbols include commodities (CL=F, GC=F), currencies (EURUSD=X), "
            "ETFs (SPY, EEM), and more."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Ticker symbol (e.g. CL=F, GC=F, SPY, EURUSD=X).",
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date (YYYY-MM-DD). Defaults to 30 days ago.",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date (YYYY-MM-DD). Defaults to today.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max rows to return. Default 30.",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_correlations",
        "description": (
            "Compute Pearson correlation between geopolitical event metrics "
            "(goldstein_mean, mentions_total, avg_tone, conflict_count, cooperation_count) "
            "and daily log returns for a symbol. Shows how events relate to market moves."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Ticker symbol to analyze.",
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date (YYYY-MM-DD). Defaults to 2016-01-01.",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date (YYYY-MM-DD). Defaults to today.",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_top_correlations",
        "description": (
            "Get the strongest event-market correlations across all 33 tracked symbols. "
            "Uses precomputed cache for fast results. Shows which symbol-metric pairs "
            "have the strongest statistical relationship."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of top correlations to return. Default 20.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_historical_patterns",
        "description": (
            "Analyze historical frequency patterns: 'When X event happens, Y symbol "
            "goes UP Z% of the time.' Checks all event groups (verbal/material cooperation, "
            "verbal/material/violent conflict) and country-specific patterns."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Ticker symbol to analyze.",
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date (YYYY-MM-DD). Defaults to 2016-01-01.",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date (YYYY-MM-DD). Defaults to today.",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "run_prediction",
        "description": (
            "Run a logistic regression prediction for a symbol's next-day market "
            "direction (UP/DOWN). Uses 7 event-based features trained on historical data. "
            "Returns probability, accuracy, and key feature contributions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Ticker symbol to predict.",
                },
                "start_date": {
                    "type": "string",
                    "description": "Training start date (YYYY-MM-DD). Defaults to 2016-01-01.",
                },
                "end_date": {
                    "type": "string",
                    "description": "Training end date (YYYY-MM-DD). Defaults to today.",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "detect_anomalies",
        "description": (
            "Detect unusual market-event mismatches using Isolation Forest, Z-scores, "
            "and domain rules. Finds unexplained moves (big return, no events), "
            "muted responses (big event, small return), and statistical outliers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Ticker symbol to analyze.",
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date (YYYY-MM-DD). Defaults to 90 days ago.",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date (YYYY-MM-DD). Defaults to today.",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "list_symbols",
        "description": (
            "List all 33 tracked financial instruments organized by category: "
            "commodities, currencies, ETFs, volatility, bonds. Includes ticker symbols "
            "and human-readable names."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_symbol_countries",
        "description": (
            "Get the country-to-symbol sensitivity mappings. Shows which countries "
            "affect which financial instruments (e.g., Russia → Ruble, Oil, Natural Gas)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Optional: get countries for a specific symbol.",
                },
            },
            "required": [],
        },
    },
]


# =============================================================================
# TOOL EXECUTION DISPATCH
# =============================================================================

def _parse_date(s: str | None, default_days_ago: int = 30) -> date:
    """Parse a YYYY-MM-DD string or return a default date."""
    if s:
        return date.fromisoformat(s)
    return date.today() - timedelta(days=default_days_ago)


def _parse_analysis_start(s: str | None) -> date:
    """Parse start date for analysis tools, defaulting to 2016-01-01."""
    if s:
        return date.fromisoformat(s)
    return date(2016, 1, 1)


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """
    Execute a tool by name and return the JSON result string.

    This is the central dispatch that maps tool names to Python functions.
    """
    try:
        result = _dispatch(tool_name, tool_input)
        return json.dumps(result, default=str)
    except Exception as e:
        logger.exception(f"Tool execution error: {tool_name}")
        return json.dumps({"error": str(e)})


def _dispatch(tool_name: str, inp: dict) -> dict | list:
    """Route tool calls to the appropriate function."""

    if tool_name == "get_recent_events":
        return _exec_get_recent_events(inp)
    elif tool_name == "get_event_summary":
        return _exec_get_event_summary(inp)
    elif tool_name == "get_market_data":
        return _exec_get_market_data(inp)
    elif tool_name == "get_correlations":
        return _exec_get_correlations(inp)
    elif tool_name == "get_top_correlations":
        return _exec_get_top_correlations(inp)
    elif tool_name == "get_historical_patterns":
        return _exec_get_historical_patterns(inp)
    elif tool_name == "run_prediction":
        return _exec_run_prediction(inp)
    elif tool_name == "detect_anomalies":
        return _exec_detect_anomalies(inp)
    elif tool_name == "list_symbols":
        return _exec_list_symbols(inp)
    elif tool_name == "get_symbol_countries":
        return _exec_get_symbol_countries(inp)
    else:
        return {"error": f"Unknown tool: {tool_name}"}


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================

def _exec_get_recent_events(inp: dict) -> list[dict]:
    start = _parse_date(inp.get("start_date"), default_days_ago=30)
    end = _parse_date(inp.get("end_date"), default_days_ago=0)
    country = inp.get("country_code")
    min_mentions = inp.get("min_mentions", 200)
    limit = inp.get("limit", 20)

    with get_session() as session:
        events = get_events_by_date_range(
            session, start, end,
            country_code=country,
            min_mentions=min_mentions,
        )

    return [
        {
            "event_date": str(e.event_date),
            "actor1_name": e.actor1_name,
            "actor1_country_code": e.actor1_country_code,
            "actor2_name": e.actor2_name,
            "actor2_country_code": e.actor2_country_code,
            "event_root_code": e.event_root_code,
            "goldstein_scale": e.goldstein_scale,
            "num_mentions": e.num_mentions,
            "avg_tone": e.avg_tone,
            "action_geo_name": e.action_geo_name,
            "action_geo_country_code": e.action_geo_country_code,
        }
        for e in events[:limit]
    ]


def _exec_get_event_summary(inp: dict) -> dict:
    start = _parse_date(inp.get("start_date"), default_days_ago=30)
    end = _parse_date(inp.get("end_date"), default_days_ago=0)
    group_by = inp.get("group_by", "country")

    with get_session() as session:
        if group_by == "type":
            rows = get_event_counts_by_type(session, start, end)
            return {
                "group_by": "type",
                "start_date": str(start),
                "end_date": str(end),
                "counts": [
                    {
                        "event_root_code": code,
                        "category": CAMEO_CATEGORIES.get(str(code).zfill(2), "other"),
                        "count": count,
                    }
                    for code, count in rows
                ],
            }
        else:
            rows = get_event_counts_by_country(session, start, end)
            return {
                "group_by": "country",
                "start_date": str(start),
                "end_date": str(end),
                "counts": [
                    {"country_code": code, "count": count}
                    for code, count in rows[:30]
                ],
            }


def _exec_get_market_data(inp: dict) -> list[dict]:
    symbol = inp["symbol"]
    start = _parse_date(inp.get("start_date"), default_days_ago=30)
    end = _parse_date(inp.get("end_date"), default_days_ago=0)
    limit = inp.get("limit", 30)

    with get_session() as session:
        rows = get_market_data(session, symbol, start, end)

    return [
        {
            "date": str(r.date),
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume,
            "daily_return": r.daily_return,
        }
        for r in rows[-limit:]
    ]


def _exec_get_correlations(inp: dict) -> list[dict]:
    symbol = inp["symbol"]
    start = _parse_analysis_start(inp.get("start_date"))
    end = _parse_date(inp.get("end_date"), default_days_ago=0)

    analyzer = CorrelationAnalyzer()
    results = analyzer.compute_correlations(symbol, start, end)

    return [
        {
            "event_metric": r.event_metric,
            "correlation": round(r.correlation, 4),
            "p_value": round(r.p_value, 4),
            "n_observations": r.n_observations,
            "significant": r.p_value < 0.05,
        }
        for r in results
    ]


def _exec_get_top_correlations(inp: dict) -> list[dict]:
    limit = inp.get("limit", 20)

    with get_session() as session:
        cached = get_cached_correlations(session, limit=limit)

    if cached:
        return cached

    # Fallback: compute live (slower)
    analyzer = CorrelationAnalyzer()
    start = date(2016, 1, 1)
    end = date.today()
    df = analyzer.top_correlated_pairs(get_all_symbols(), start, end, limit=limit)

    if df.empty:
        return []

    return df.to_dict(orient="records")


def _exec_get_historical_patterns(inp: dict) -> list[dict]:
    symbol = inp["symbol"]
    start = _parse_analysis_start(inp.get("start_date"))
    end = _parse_date(inp.get("end_date"), default_days_ago=0)

    analyzer = HistoricalPatternAnalyzer()
    patterns = analyzer.all_patterns_for_symbol(symbol, start, end)

    return [
        {
            "event_filter": p.event_filter,
            "total_occurrences": p.total_occurrences,
            "up_count": p.up_count,
            "down_count": p.down_count,
            "up_percentage": round(p.up_percentage, 1),
            "avg_return_all": round(p.avg_return_all, 4),
            "p_value": round(p.p_value, 4),
            "is_significant": p.p_value < 0.05,
        }
        for p in patterns
    ]


def _exec_run_prediction(inp: dict) -> dict:
    symbol = inp["symbol"]
    start = _parse_analysis_start(inp.get("start_date"))
    end = _parse_date(inp.get("end_date"), default_days_ago=0)

    analyzer = LogisticRegressionAnalyzer()
    summary = analyzer.get_model_summary(symbol, start, end)

    if not summary:
        return {"error": f"Insufficient data for prediction on {symbol}"}

    # Also run a prediction with recent conditions (average features)
    result = analyzer.train_and_predict(
        symbol, start, end,
        current_features={
            "goldstein_mean": 0.0,
            "goldstein_min": -2.0,
            "goldstein_max": 5.0,
            "mentions_total": 500,
            "avg_tone": -1.0,
            "conflict_count": 5,
            "cooperation_count": 10,
        },
    )

    if not result:
        return {"error": f"Prediction failed for {symbol}"}

    return {
        "symbol": symbol,
        "prediction": result.prediction,
        "probability_up": round(result.probability_up, 3),
        "accuracy": round(result.accuracy, 3),
        "n_training_samples": result.n_training_samples,
        "top_features": result.feature_contributions[:5],
        "disclaimer": (
            "This is a statistical model for educational purposes only. "
            "Not financial advice."
        ),
    }


def _exec_detect_anomalies(inp: dict) -> dict:
    symbol = inp["symbol"]
    start = _parse_date(inp.get("start_date"), default_days_ago=90)
    end = _parse_date(inp.get("end_date"), default_days_ago=0)

    detector = ProductionAnomalyDetector()
    anomalies = detector.detect_all(symbol, start, end)
    report = detector.get_anomaly_report(anomalies, symbol, start, end)

    return {
        "symbol": symbol,
        "period": f"{start} to {end}",
        "total_anomalies": report.anomaly_count,
        "anomaly_rate": round(report.anomaly_rate * 100, 1),
        "breakdown": {
            "unexplained_moves": report.unexplained_moves,
            "muted_responses": report.muted_responses,
            "statistical_outliers": report.statistical_outliers,
        },
        "top_anomalies": [
            {
                "date": str(a.date),
                "type": a.anomaly_type,
                "actual_return_pct": round(a.actual_return * 100, 2),
                "z_score": round(a.z_score, 2),
                "anomaly_probability": round(a.anomaly_probability, 2),
                "detected_by": a.detected_by,
            }
            for a in report.top_anomalies[:5]
        ],
    }


def _exec_list_symbols(inp: dict) -> dict:
    result = {}
    for category, symbols in SYMBOLS.items():
        result[category] = [
            {"symbol": sym, "name": name}
            for sym, name in symbols.items()
        ]
    return result


def _exec_get_symbol_countries(inp: dict) -> dict:
    symbol = inp.get("symbol")
    if symbol:
        countries = SYMBOL_COUNTRY_MAP.get(symbol, [])
        info = get_symbol_info(symbol)
        return {
            "symbol": symbol,
            "name": info["name"] if info else symbol,
            "sensitive_to_countries": countries,
        }

    # Return full mapping
    return {
        "country_to_assets": {
            country: assets
            for country, assets in COUNTRY_ASSET_MAP.items()
        },
        "symbol_to_countries": {
            symbol: countries
            for symbol, countries in SYMBOL_COUNTRY_MAP.items()
        },
    }
