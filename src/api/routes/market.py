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
from sqlalchemy import func

from src.db.connection import get_session
from src.db.models import MarketData
from src.config.constants import SYMBOLS, get_all_symbols, get_symbol_info
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
        start_date = end_date - timedelta(days=30)

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
