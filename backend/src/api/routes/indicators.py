"""
Economic Indicators API Router.

Endpoints for querying FRED economic indicator data.

USAGE:
------
    GET /api/indicators/latest - Latest value + delta for each series
    GET /api/indicators/{series_id} - Time series for a single indicator
"""

from datetime import date, timedelta
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.db.connection import get_session
from src.db.queries import (
    get_indicators_with_deltas,
    get_indicators_by_series,
)
from src.api.schemas import IndicatorResponse, IndicatorWithDeltaResponse

router = APIRouter(prefix="/indicators", tags=["Indicators"])


def get_db():
    """Dependency to get database session."""
    with get_session() as session:
        yield session


@router.get("/latest", response_model=list[IndicatorWithDeltaResponse])
def latest_indicators(db: Session = Depends(get_db)):
    """
    Get the latest value for each FRED series with change from previous observation.

    Returns current value, previous value, absolute change, and percent change
    for each tracked indicator (GDP, CPI, unemployment, Fed rate, 10Y yield,
    consumer sentiment).
    """
    return get_indicators_with_deltas(db)


@router.get("/{series_id}", response_model=list[IndicatorResponse])
def indicator_series(
    series_id: str,
    start_date: date | None = Query(None, description="Start of date range"),
    end_date: date | None = Query(None, description="End of date range"),
    db: Session = Depends(get_db),
):
    """
    Get the time series for a single FRED indicator.

    **Examples:**
    - Federal Funds Rate: `/api/indicators/DFF`
    - CPI last year: `/api/indicators/CPIAUCSL?start_date=2025-03-01`
    - GDP all time: `/api/indicators/GDP`
    """
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=365)

    return get_indicators_by_series(db, series_id, start_date, end_date)
