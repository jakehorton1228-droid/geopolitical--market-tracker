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

from src.db.models import (
    Event, MarketData, AnalysisResult, EventMarketLink,
    NewsHeadline, EconomicIndicator, PredictionMarket,
)
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

# =============================================================================
# NEWS HEADLINE QUERIES
# =============================================================================


def upsert_headline(session: Session, headline_data: dict) -> NewsHeadline | None:
    """
    Insert a headline if its URL doesn't already exist. Skip if it does.

    Unlike upsert_event() which UPDATES existing rows, headlines don't change
    after publication — so we just skip duplicates entirely.

    Args:
        session: Database session
        headline_data: Dict with keys matching NewsHeadline columns

    Returns:
        The new NewsHeadline object, or None if it already existed
    """
    # Check if we already have this URL
    existing = session.query(NewsHeadline).filter(
        NewsHeadline.url == headline_data["url"]
    ).first()

    if existing:
        return None  # Already ingested, skip

    headline = NewsHeadline(**headline_data)
    session.add(headline)
    return headline


def get_headlines_by_date_range(
    session: Session,
    start_date: date,
    end_date: date,
    source: str | None = None,
) -> list[NewsHeadline]:
    """
    Fetch headlines within a date range, optionally filtered by source.

    Mirrors get_events_by_date_range() but for news headlines.
    Note: published_at is DateTime, so we compare against the date portion.

    Args:
        session: Database session
        start_date: Start of range (inclusive)
        end_date: End of range (inclusive)
        source: Filter by source name ("reuters", "ap", "bbc", "aljazeera")

    Returns:
        List of NewsHeadline objects, newest first
    """
    query = session.query(NewsHeadline).filter(
        func.date(NewsHeadline.published_at) >= start_date,
        func.date(NewsHeadline.published_at) <= end_date,
    )

    if source:
        query = query.filter(NewsHeadline.source == source)

    return query.order_by(NewsHeadline.published_at.desc()).all()


def get_latest_headline_date(session: Session, source: str | None = None) -> date | None:
    """
    Get the most recent headline date, optionally for a specific source.

    Used by the ingestion module to avoid re-fetching headlines we already have.
    Mirrors get_latest_market_date().

    Args:
        session: Database session
        source: Optional source filter

    Returns:
        The date of the most recent headline, or None if table is empty
    """
    query = session.query(func.max(NewsHeadline.published_at))

    if source:
        query = query.filter(NewsHeadline.source == source)

    result = query.scalar()
    # published_at is DateTime, return just the date portion
    return result.date() if result else None


# =============================================================================
# ECONOMIC INDICATOR QUERIES
# =============================================================================


def upsert_economic_indicator(
    session: Session, indicator_data: dict
) -> EconomicIndicator | None:
    """
    Insert an economic indicator observation if it doesn't already exist.

    Like headlines, FRED observations don't change after publication (revisions
    get new dates), so we skip duplicates entirely.

    Args:
        session: Database session
        indicator_data: Dict with series_id, series_name, date, value

    Returns:
        The new EconomicIndicator object, or None if it already existed
    """
    existing = session.query(EconomicIndicator).filter(
        EconomicIndicator.series_id == indicator_data["series_id"],
        EconomicIndicator.date == indicator_data["date"],
    ).first()

    if existing:
        return None  # Already ingested, skip

    indicator = EconomicIndicator(**indicator_data)
    session.add(indicator)
    return indicator


def get_indicators_by_series(
    session: Session,
    series_id: str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[EconomicIndicator]:
    """
    Fetch observations for a FRED series within an optional date range.

    Args:
        session: Database session
        series_id: FRED series ID (e.g., "GDP", "UNRATE")
        start_date: Start of range (inclusive)
        end_date: End of range (inclusive)

    Returns:
        List of EconomicIndicator objects, oldest first
    """
    query = session.query(EconomicIndicator).filter(
        EconomicIndicator.series_id == series_id
    )

    if start_date:
        query = query.filter(EconomicIndicator.date >= start_date)
    if end_date:
        query = query.filter(EconomicIndicator.date <= end_date)

    return query.order_by(EconomicIndicator.date).all()


def get_latest_indicator_date(
    session: Session, series_id: str
) -> date | None:
    """
    Get the most recent observation date for a FRED series.

    Used by the ingestion module to avoid re-fetching data we already have.

    Args:
        session: Database session
        series_id: FRED series ID

    Returns:
        The date of the most recent observation, or None if no data exists
    """
    return session.query(func.max(EconomicIndicator.date)).filter(
        EconomicIndicator.series_id == series_id
    ).scalar()


def get_all_latest_indicators(session: Session) -> list[EconomicIndicator]:
    """
    Get the most recent observation for each FRED series.

    Useful for dashboard display — shows current values of all tracked indicators.

    Returns:
        List of EconomicIndicator objects (one per series)
    """
    # Subquery to find the max date per series
    subquery = session.query(
        EconomicIndicator.series_id,
        func.max(EconomicIndicator.date).label("max_date"),
    ).group_by(EconomicIndicator.series_id).subquery()

    return session.query(EconomicIndicator).join(
        subquery,
        and_(
            EconomicIndicator.series_id == subquery.c.series_id,
            EconomicIndicator.date == subquery.c.max_date,
        ),
    ).all()


# =============================================================================
# PREDICTION MARKET QUERIES
# =============================================================================


def upsert_prediction_market(
    session: Session, market_data: dict
) -> PredictionMarket | None:
    """
    Insert a prediction market snapshot if we don't already have one for today.

    We store one snapshot per market per day. If we already have today's snapshot,
    we skip it (probabilities within a single day are close enough).

    Args:
        session: Database session
        market_data: Dict with market_id, question, yes_price, snapshot_at, etc.

    Returns:
        The new PredictionMarket object, or None if today's snapshot exists
    """
    snapshot_date = market_data["snapshot_at"].date()

    existing = session.query(PredictionMarket).filter(
        PredictionMarket.market_id == market_data["market_id"],
        PredictionMarket.snapshot_date == snapshot_date,
    ).first()

    if existing:
        return None  # Already have today's snapshot

    market_data["snapshot_date"] = snapshot_date
    market = PredictionMarket(**market_data)
    session.add(market)
    return market


def get_market_snapshots(
    session: Session,
    market_id: str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[PredictionMarket]:
    """
    Fetch daily snapshots for a specific prediction market.

    Returns the probability time series for a market, useful for
    plotting how crowd sentiment shifts over time.

    Args:
        session: Database session
        market_id: Polymarket's market ID
        start_date: Start of range (inclusive)
        end_date: End of range (inclusive)

    Returns:
        List of PredictionMarket objects, oldest first
    """
    query = session.query(PredictionMarket).filter(
        PredictionMarket.market_id == market_id
    )

    if start_date:
        query = query.filter(PredictionMarket.snapshot_date >= start_date)
    if end_date:
        query = query.filter(PredictionMarket.snapshot_date <= end_date)

    return query.order_by(PredictionMarket.snapshot_date).all()


def get_latest_predictions(
    session: Session, limit: int = 50
) -> list[PredictionMarket]:
    """
    Get the most recent snapshot for each tracked market.

    Useful for dashboard display — shows current probabilities for
    all geopolitical prediction markets.

    Args:
        session: Database session
        limit: Maximum number of markets to return

    Returns:
        List of PredictionMarket objects (one per market), sorted by volume
    """
    subquery = session.query(
        PredictionMarket.market_id,
        func.max(PredictionMarket.snapshot_date).label("max_date"),
    ).group_by(PredictionMarket.market_id).subquery()

    return session.query(PredictionMarket).join(
        subquery,
        and_(
            PredictionMarket.market_id == subquery.c.market_id,
            PredictionMarket.snapshot_date == subquery.c.max_date,
        ),
    ).order_by(
        PredictionMarket.volume_24h.desc()
    ).limit(limit).all()