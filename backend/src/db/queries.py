"""Database query functions (repository pattern).

This module provides clean, reusable query functions that abstract
database operations. This pattern keeps SQL/ORM logic separate from
business logic.

Learning note: The repository pattern is widely used in industry because:
1. It centralizes data access logic
2. Makes testing easier (you can mock the repository)
3. Allows changing the database without touching business code
"""

from datetime import date, timedelta
from sqlalchemy import func, and_, or_
from sqlalchemy.orm import Session

from src.db.models import Event, MarketData, AnalysisResult, EventMarketLink
from src.config.constants import get_event_group


# =============================================================================
# EVENT QUERIES
# =============================================================================

def get_events_by_date_range(
    session: Session,
    start_date: date,
    end_date: date,
    country_code: str | None = None,
    event_root_codes: list[str] | None = None,
    min_goldstein: float | None = None,
    min_mentions: int | None = None,
) -> list[Event]:
    """
    Fetch events within a date range with optional filters.

    Args:
        session: Database session
        start_date: Start of date range (inclusive)
        end_date: End of date range (inclusive)
        country_code: Filter by country (3-letter ISO code)
        event_root_codes: Filter by CAMEO root codes (e.g., ["18", "19", "20"])
        min_goldstein: Minimum absolute Goldstein score
        min_mentions: Minimum number of mentions

    Returns:
        List of Event objects matching criteria
    """
    query = session.query(Event).filter(
        Event.event_date >= start_date,
        Event.event_date <= end_date,
    )

    if country_code:
        query = query.filter(
            or_(
                Event.actor1_country_code == country_code,
                Event.actor2_country_code == country_code,
                Event.action_geo_country_code == country_code,
            )
        )

    if event_root_codes:
        query = query.filter(Event.event_root_code.in_(event_root_codes))

    if min_goldstein is not None:
        query = query.filter(func.abs(Event.goldstein_scale) >= min_goldstein)

    if min_mentions is not None:
        query = query.filter(Event.num_mentions >= min_mentions)

    return query.order_by(Event.event_date.desc(), Event.num_mentions.desc()).all()


def get_significant_events(
    session: Session,
    start_date: date,
    end_date: date,
    min_mentions: int = 10,
    min_goldstein: float = 3.0,
) -> list[Event]:
    """
    Fetch events that are likely market-moving.

    Significant events have high media coverage and strong conflict/cooperation scores.
    """
    return get_events_by_date_range(
        session,
        start_date,
        end_date,
        min_mentions=min_mentions,
        min_goldstein=min_goldstein,
    )


def get_conflict_events(
    session: Session,
    start_date: date,
    end_date: date,
) -> list[Event]:
    """Fetch violent conflict events (CAMEO codes 18, 19, 20)."""
    return get_events_by_date_range(
        session,
        start_date,
        end_date,
        event_root_codes=["18", "19", "20"],
    )


def get_events_for_country(
    session: Session,
    country_code: str,
    days_back: int = 30,
) -> list[Event]:
    """Fetch recent events involving a specific country."""
    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)
    return get_events_by_date_range(
        session,
        start_date,
        end_date,
        country_code=country_code,
    )


def upsert_event(session: Session, event_data: dict) -> Event:
    """
    Insert or update an event by global_event_id.

    Args:
        session: Database session
        event_data: Dictionary with event fields

    Returns:
        The created or updated Event object
    """
    global_event_id = event_data.get("global_event_id")
    existing = session.query(Event).filter(
        Event.global_event_id == global_event_id
    ).first()

    if existing:
        for key, value in event_data.items():
            if hasattr(existing, key):
                setattr(existing, key, value)
        return existing
    else:
        event = Event(**event_data)
        session.add(event)
        return event


# =============================================================================
# MARKET DATA QUERIES
# =============================================================================

def get_market_data(
    session: Session,
    symbol: str,
    start_date: date,
    end_date: date,
) -> list[MarketData]:
    """Fetch market data for a symbol within a date range."""
    return session.query(MarketData).filter(
        MarketData.symbol == symbol,
        MarketData.date >= start_date,
        MarketData.date <= end_date,
    ).order_by(MarketData.date).all()


def get_market_data_for_symbols(
    session: Session,
    symbols: list[str],
    start_date: date,
    end_date: date,
) -> list[MarketData]:
    """Fetch market data for multiple symbols within a date range."""
    return session.query(MarketData).filter(
        MarketData.symbol.in_(symbols),
        MarketData.date >= start_date,
        MarketData.date <= end_date,
    ).order_by(MarketData.symbol, MarketData.date).all()


def get_latest_market_date(session: Session, symbol: str) -> date | None:
    """Get the most recent date we have data for a symbol."""
    result = session.query(func.max(MarketData.date)).filter(
        MarketData.symbol == symbol
    ).scalar()
    return result


def upsert_market_data(session: Session, data: dict) -> MarketData:
    """
    Insert or update market data by symbol and date.

    Args:
        session: Database session
        data: Dictionary with market data fields

    Returns:
        The created or updated MarketData object
    """
    existing = session.query(MarketData).filter(
        MarketData.symbol == data["symbol"],
        MarketData.date == data["date"],
    ).first()

    if existing:
        for key, value in data.items():
            if hasattr(existing, key):
                setattr(existing, key, value)
        return existing
    else:
        market_data = MarketData(**data)
        session.add(market_data)
        return market_data


def bulk_insert_market_data(session: Session, data_list: list[dict]) -> int:
    """
    Bulk insert market data (faster than individual inserts).

    Note: This does NOT handle duplicates - use only for new data.

    Returns:
        Number of rows inserted
    """
    if not data_list:
        return 0

    session.bulk_insert_mappings(MarketData, data_list)
    return len(data_list)


# =============================================================================
# ANALYSIS RESULT QUERIES
# =============================================================================

def get_analysis_results_for_event(
    session: Session,
    event_id: int,
) -> list[AnalysisResult]:
    """Get all analysis results for a specific event."""
    return session.query(AnalysisResult).filter(
        AnalysisResult.event_id == event_id
    ).all()


def get_significant_results(
    session: Session,
    start_date: date | None = None,
    end_date: date | None = None,
    min_car: float | None = None,
) -> list[AnalysisResult]:
    """
    Fetch statistically significant analysis results.

    Args:
        start_date: Filter by event date start
        end_date: Filter by event date end
        min_car: Minimum absolute CAR value

    Returns:
        List of significant AnalysisResult objects
    """
    query = session.query(AnalysisResult).join(Event).filter(
        AnalysisResult.is_significant == True
    )

    if start_date:
        query = query.filter(Event.event_date >= start_date)
    if end_date:
        query = query.filter(Event.event_date <= end_date)
    if min_car is not None:
        query = query.filter(func.abs(AnalysisResult.car) >= min_car)

    return query.order_by(func.abs(AnalysisResult.car).desc()).all()


def get_anomalies(
    session: Session,
    anomaly_type: str | None = None,
    limit: int = 100,
) -> list[AnalysisResult]:
    """
    Fetch detected anomalies.

    Args:
        anomaly_type: Filter by type ("unexplained_move", "muted_response", etc.)
        limit: Maximum results to return

    Returns:
        List of anomaly AnalysisResult objects
    """
    query = session.query(AnalysisResult).filter(
        AnalysisResult.is_anomaly == True
    )

    if anomaly_type:
        query = query.filter(AnalysisResult.anomaly_type == anomaly_type)

    return query.order_by(
        AnalysisResult.anomaly_score.desc()
    ).limit(limit).all()


def save_analysis_result(session: Session, result: dict) -> AnalysisResult:
    """Save an analysis result, replacing if exists."""
    existing = session.query(AnalysisResult).filter(
        AnalysisResult.event_id == result["event_id"],
        AnalysisResult.symbol == result["symbol"],
        AnalysisResult.analysis_type == result["analysis_type"],
    ).first()

    if existing:
        for key, value in result.items():
            if hasattr(existing, key):
                setattr(existing, key, value)
        return existing
    else:
        analysis = AnalysisResult(**result)
        session.add(analysis)
        return analysis


# =============================================================================
# AGGREGATE QUERIES
# =============================================================================

def get_event_counts_by_country(
    session: Session,
    start_date: date,
    end_date: date,
) -> list[tuple[str, int]]:
    """
    Count events by country within a date range.

    Returns list of (country_code, count) tuples ordered by count descending.
    """
    return session.query(
        Event.action_geo_country_code,
        func.count(Event.id).label("count")
    ).filter(
        Event.event_date >= start_date,
        Event.event_date <= end_date,
        Event.action_geo_country_code.isnot(None),
    ).group_by(
        Event.action_geo_country_code
    ).order_by(
        func.count(Event.id).desc()
    ).all()


def get_event_counts_by_type(
    session: Session,
    start_date: date,
    end_date: date,
) -> list[tuple[str, int]]:
    """
    Count events by CAMEO root code within a date range.

    Returns list of (event_root_code, count) tuples.
    """
    return session.query(
        Event.event_root_code,
        func.count(Event.id).label("count")
    ).filter(
        Event.event_date >= start_date,
        Event.event_date <= end_date,
    ).group_by(
        Event.event_root_code
    ).order_by(
        Event.event_root_code
    ).all()


def get_average_market_reaction(
    session: Session,
    event_root_code: str,
    symbol: str,
) -> dict | None:
    """
    Calculate average market reaction for an event type and symbol.

    Returns dict with avg_car, count, and significance rate.
    """
    results = session.query(
        func.avg(AnalysisResult.car).label("avg_car"),
        func.count(AnalysisResult.id).label("count"),
        func.avg(
            func.cast(AnalysisResult.is_significant, Integer)
        ).label("significance_rate"),
    ).join(Event).filter(
        Event.event_root_code == event_root_code,
        AnalysisResult.symbol == symbol,
        AnalysisResult.analysis_type == "event_study",
    ).first()

    if results and results.count > 0:
        return {
            "avg_car": float(results.avg_car) if results.avg_car else 0,
            "count": results.count,
            "significance_rate": float(results.significance_rate) if results.significance_rate else 0,
        }
    return None


# =============================================================================
# CORRELATION CACHE
# =============================================================================


def save_correlation_cache(
    session: Session,
    results: list[dict],
    method: str = "pearson",
) -> int:
    """
    Replace cached correlations with fresh results.

    Deletes all existing rows for the given method and bulk inserts new ones.
    Returns the number of rows inserted.
    """
    from src.db.models import CorrelationCache

    # Clear existing cache for this method
    session.query(CorrelationCache).filter(
        CorrelationCache.method == method,
    ).delete()

    # Insert new rows
    for r in results:
        session.add(CorrelationCache(
            symbol=r["symbol"],
            event_metric=r["event_metric"],
            correlation=r["correlation"],
            p_value=r["p_value"],
            n_observations=r["n_observations"],
            method=method,
            start_date=r["start_date"],
            end_date=r["end_date"],
        ))

    session.flush()
    return len(results)


def get_cached_correlations(
    session: Session,
    method: str = "pearson",
    limit: int = 20,
) -> list[dict]:
    """
    Get top cached correlations sorted by absolute correlation strength.

    Returns list of dicts matching the /api/correlation/top response shape.
    Returns empty list if cache is empty.
    """
    from src.db.models import CorrelationCache

    rows = session.query(CorrelationCache).filter(
        CorrelationCache.method == method,
    ).order_by(
        func.abs(CorrelationCache.correlation).desc(),
    ).limit(limit).all()

    return [
        {
            "symbol": r.symbol,
            "event_metric": r.event_metric,
            "correlation": r.correlation,
            "abs_correlation": abs(r.correlation),
            "p_value": r.p_value,
            "n_observations": r.n_observations,
            "direction": "positive" if r.correlation > 0 else "negative",
            "cached": True,
            "computed_at": r.computed_at.isoformat() if r.computed_at else None,
        }
        for r in rows
    ]
