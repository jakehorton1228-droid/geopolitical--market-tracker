"""
Market Data API Router.

Endpoints for querying financial market data.

USAGE:
------
    GET /api/market - List market data with filters
    GET /api/market/symbols - Get available symbols
    GET /api/market/{symbol} - Get data for specific symbol
    GET /api/market/{symbol}/latest - Get latest price for symbol
"""

from datetime import date, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import case, func, or_, text

from src.db.connection import get_session
from src.db.models import MarketData, Event
from src.config.constants import SYMBOLS, get_all_symbols, get_symbol_info, SYMBOL_COUNTRY_MAP, EVENT_GROUPS
from src.api.schemas import MarketDataResponse

router = APIRouter(prefix="/market", tags=["Market Data"])


def get_db():
    """Dependency to get database session."""
    with get_session() as session:
        yield session


@router.get("", response_model=list[MarketDataResponse])
def list_market_data(
    symbol: str | None = Query(None, description="Single symbol (e.g., CL=F)"),
    symbols: str | None = Query(None, description="Comma-separated symbols"),
    start_date: date | None = Query(None, description="Start date"),
    end_date: date | None = Query(None, description="End date"),
    limit: int = Query(1000, ge=1, le=10000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """
    List market data with optional filters.

    **Examples:**
    - Get oil prices: `?symbol=CL=F`
    - Get multiple assets: `?symbols=CL=F,GC=F,SPY`
    - Get recent data: `?start_date=2024-01-01`
    """
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=365)

    query = db.query(MarketData).filter(
        MarketData.date >= start_date,
        MarketData.date <= end_date,
    )

    # Filter by symbol(s)
    if symbol:
        query = query.filter(MarketData.symbol == symbol)
    elif symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]
        query = query.filter(MarketData.symbol.in_(symbol_list))

    data = query.order_by(
        MarketData.symbol,
        MarketData.date.desc(),
    ).offset(offset).limit(limit).all()

    return data


@router.get("/symbols", response_model=dict)
def list_symbols():
    """
    Get all available market symbols organized by category.

    Returns commodities, currencies, ETFs, volatility, and bonds.
    """
    return {
        "symbols": SYMBOLS,
        "total": len(get_all_symbols()),
    }


@router.get("/symbols/flat", response_model=list[dict])
def list_symbols_flat():
    """Get all symbols as a flat list with metadata."""
    result = []
    for symbol in get_all_symbols():
        info = get_symbol_info(symbol)
        if info:
            result.append(info)
    return result


@router.get("/{symbol}", response_model=list[MarketDataResponse])
def get_symbol_data(
    symbol: str,
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    limit: int = Query(365, ge=1, le=10000),
    db: Session = Depends(get_db),
):
    """
    Get market data for a specific symbol.

    **Examples:**
    - Get oil prices: `/api/market/CL=F`
    - Get gold last 90 days: `/api/market/GC=F?start_date=2024-01-01`
    """
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=365)

    data = db.query(MarketData).filter(
        MarketData.symbol == symbol,
        MarketData.date >= start_date,
        MarketData.date <= end_date,
    ).order_by(MarketData.date.desc()).limit(limit).all()

    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for symbol '{symbol}'. Check /api/market/symbols for available symbols.",
        )

    return data


@router.get("/{symbol}/latest", response_model=MarketDataResponse)
def get_latest_price(
    symbol: str,
    db: Session = Depends(get_db),
):
    """Get the most recent price for a symbol."""
    data = db.query(MarketData).filter(
        MarketData.symbol == symbol,
    ).order_by(MarketData.date.desc()).first()

    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for symbol '{symbol}'",
        )

    return data


@router.get("/{symbol}/returns", response_model=list[dict])
def get_symbol_returns(
    symbol: str,
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    db: Session = Depends(get_db),
):
    """
    Get daily returns for a symbol.

    Returns date and daily_return (as percentage).
    """
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=90)

    data = db.query(
        MarketData.date,
        MarketData.daily_return,
    ).filter(
        MarketData.symbol == symbol,
        MarketData.date >= start_date,
        MarketData.date <= end_date,
        MarketData.daily_return.isnot(None),
    ).order_by(MarketData.date).all()

    if not data:
        raise HTTPException(status_code=404, detail=f"No return data for '{symbol}'")

    return [
        {"date": d.date, "return_pct": d.daily_return * 100}
        for d in data
    ]


@router.get("/{symbol}/stats", response_model=dict)
def get_symbol_stats(
    symbol: str,
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    db: Session = Depends(get_db),
):
    """
    Get summary statistics for a symbol.

    Returns mean return, volatility, min, max, and count.
    """
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=90)

    stats = db.query(
        func.count(MarketData.id).label("count"),
        func.avg(MarketData.daily_return).label("mean_return"),
        func.min(MarketData.daily_return).label("min_return"),
        func.max(MarketData.daily_return).label("max_return"),
        func.min(MarketData.close).label("min_price"),
        func.max(MarketData.close).label("max_price"),
    ).filter(
        MarketData.symbol == symbol,
        MarketData.date >= start_date,
        MarketData.date <= end_date,
    ).first()

    if not stats or stats.count == 0:
        raise HTTPException(status_code=404, detail=f"No data for '{symbol}'")

    # Get symbol info
    info = get_symbol_info(symbol)

    return {
        "symbol": symbol,
        "name": info["name"] if info else symbol,
        "category": info["category"] if info else "unknown",
        "start_date": start_date,
        "end_date": end_date,
        "data_points": stats.count,
        "mean_daily_return_pct": float(stats.mean_return) * 100 if stats.mean_return else None,
        "min_return_pct": float(stats.min_return) * 100 if stats.min_return else None,
        "max_return_pct": float(stats.max_return) * 100 if stats.max_return else None,
        "min_price": float(stats.min_price) if stats.min_price else None,
        "max_price": float(stats.max_price) if stats.max_price else None,
    }


@router.get("/{symbol}/with-events", response_model=list[dict])
def get_symbol_with_events(
    symbol: str,
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    min_mentions: int = Query(5, ge=0, description="Min mentions to include event"),
    db: Session = Depends(get_db),
):
    """
    Get price data merged with event summaries by date.

    Powers the Event Timeline chart (price line + event scatter overlay).
    Each row contains market data plus event aggregates for that day.
    """
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=180)

    # Fetch market data
    market_rows = db.query(MarketData).filter(
        MarketData.symbol == symbol,
        MarketData.date >= start_date,
        MarketData.date <= end_date,
    ).order_by(MarketData.date).all()

    if not market_rows:
        raise HTTPException(status_code=404, detail=f"No data for '{symbol}'")

    # Build country filter for events relevant to this symbol
    relevant_countries = SYMBOL_COUNTRY_MAP.get(symbol, [])
    country_filter = []
    if relevant_countries:
        country_filter = [or_(
            Event.actor1_country_code.in_(relevant_countries),
            Event.actor2_country_code.in_(relevant_countries),
            Event.action_geo_country_code.in_(relevant_countries),
        )]

    # Conflict = codes 14-20, Cooperation = codes 01-08
    conflict_codes = EVENT_GROUPS["material_conflict"] + EVENT_GROUPS["violent_conflict"]
    cooperation_codes = (
        EVENT_GROUPS["verbal_cooperation"] + EVENT_GROUPS["material_cooperation"]
    )

    # SQL aggregation: counts, avg goldstein, mentions, conflict/cooperation per date
    agg_query = db.query(
        Event.event_date.label("date"),
        func.count(Event.id).label("event_count"),
        func.avg(Event.goldstein_scale).label("avg_goldstein"),
        func.sum(Event.num_mentions).label("total_mentions"),
        func.sum(case(
            (Event.event_root_code.in_(conflict_codes), 1), else_=0,
        )).label("conflict_count"),
        func.sum(case(
            (Event.event_root_code.in_(cooperation_codes), 1), else_=0,
        )).label("cooperation_count"),
    ).filter(
        Event.event_date >= start_date,
        Event.event_date <= end_date,
        *country_filter,
    )

    if min_mentions > 0:
        agg_query = agg_query.filter(Event.num_mentions >= min_mentions)

    agg_rows = agg_query.group_by(Event.event_date).all()

    # Build lookup: date -> aggregated event stats
    events_by_date = {}
    for row in agg_rows:
        events_by_date[row.date] = {
            "event_count": row.event_count,
            "avg_goldstein": round(float(row.avg_goldstein), 2) if row.avg_goldstein else 0,
            "total_mentions": int(row.total_mentions or 0),
            "conflict_count": int(row.conflict_count or 0),
            "cooperation_count": int(row.cooperation_count or 0),
        }

    # Top event per date (highest mentions) — use DISTINCT ON
    top_event_sql = text("""
        SELECT DISTINCT ON (event_date)
            event_date, id, action_geo_name, goldstein_scale,
            num_mentions, event_root_code
        FROM events
        WHERE event_date >= :start AND event_date <= :end
          AND event_date = ANY(:dates)
          AND (:no_country_filter OR actor1_country_code = ANY(:countries)
               OR actor2_country_code = ANY(:countries)
               OR action_geo_country_code = ANY(:countries))
        ORDER BY event_date, num_mentions DESC NULLS LAST
    """)

    event_dates = list(events_by_date.keys())
    if event_dates:
        top_rows = db.execute(top_event_sql, {
            "start": start_date,
            "end": end_date,
            "dates": event_dates,
            "countries": relevant_countries or [],
            "no_country_filter": not relevant_countries,
        }).fetchall()

        # Map event_root_code to group name
        code_to_group = {}
        for group_name, codes in EVENT_GROUPS.items():
            for code in codes:
                code_to_group[code] = group_name

        for row in top_rows:
            d = row.event_date
            if d in events_by_date:
                root_code = str(row.event_root_code).zfill(2)[:2] if row.event_root_code else ""
                events_by_date[d]["top_event"] = {
                    "id": row.id,
                    "description": row.action_geo_name or "",
                    "goldstein": row.goldstein_scale,
                    "mentions": row.num_mentions or 0,
                    "group": code_to_group.get(root_code, "other"),
                }

    # Build response: market data + event overlay
    result = []
    for m in market_rows:
        row = {
            "date": str(m.date),
            "close": float(m.close),
            "open": float(m.open) if m.open else None,
            "high": float(m.high) if m.high else None,
            "low": float(m.low) if m.low else None,
            "volume": m.volume,
            "daily_return": m.daily_return,
        }

        event_agg = events_by_date.get(m.date)
        if event_agg:
            row.update(event_agg)
        else:
            row["event_count"] = 0

        result.append(row)

    return result
