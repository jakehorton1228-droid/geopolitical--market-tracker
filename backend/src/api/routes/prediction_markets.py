"""
Prediction Markets API Router.

Endpoints for querying Polymarket prediction market data.

USAGE:
------
    GET /api/prediction-markets - All markets with latest snapshot
    GET /api/prediction-markets/movers - Biggest probability changes
    GET /api/prediction-markets/{market_id}/history - Probability time series
"""

from datetime import date, timedelta
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.db.connection import get_session
from src.db.queries import (
    get_latest_predictions,
    get_prediction_movers,
    get_market_snapshots,
)
from src.api.schemas import (
    PredictionMarketResponse,
    PredictionMoverResponse,
)

router = APIRouter(prefix="/prediction-markets", tags=["Prediction Markets"])


def get_db():
    """Dependency to get database session."""
    with get_session() as session:
        yield session


@router.get("", response_model=list[PredictionMarketResponse])
def list_markets(
    limit: int = Query(50, ge=1, le=200, description="Maximum markets to return"),
    db: Session = Depends(get_db),
):
    """
    List all tracked prediction markets with their latest snapshot.

    Returns markets sorted by 24-hour trading volume (most active first).
    """
    return get_latest_predictions(db, limit=limit)


@router.get("/movers", response_model=list[PredictionMoverResponse])
def market_movers(
    days_back: int = Query(7, ge=1, le=30, description="Compare to N days ago"),
    limit: int = Query(10, ge=1, le=50, description="Number of movers to return"),
    db: Session = Depends(get_db),
):
    """
    Get prediction markets with the biggest probability changes.

    Compares each market's current probability to its value N days ago.
    Returns markets sorted by absolute change — the "biggest movers."

    A market moving from 20% to 45% (+25 points) indicates a significant
    shift in crowd expectations — a potential leading indicator for
    financial markets.
    """
    return get_prediction_movers(db, days_back=days_back, limit=limit)


@router.get("/{market_id}/history", response_model=list[PredictionMarketResponse])
def market_history(
    market_id: str,
    start_date: date | None = Query(None, description="Start of date range"),
    end_date: date | None = Query(None, description="End of date range"),
    db: Session = Depends(get_db),
):
    """
    Get the probability time series for a specific prediction market.

    Returns daily snapshots showing how the crowd's estimated probability
    has changed over time. Useful for charting probability trends.
    """
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=90)

    return get_market_snapshots(db, market_id, start_date, end_date)
