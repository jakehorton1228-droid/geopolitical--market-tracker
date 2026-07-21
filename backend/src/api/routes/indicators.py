"""
Economic Indicators API Router.

Endpoints for querying FRED economic indicator data.

USAGE:
------
    GET /api/indicators/latest - Latest value + delta for each series
    GET /api/indicators/{series_id} - Time series for a single indicator
"""

from datetime import date
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.api.deps import get_db, DateRange
from src.db.queries import (
    get_indicators_with_deltas,
    get_indicators_by_series,
)
from src.api.schemas import IndicatorResponse, IndicatorWithDeltaResponse

router = APIRouter(prefix="/indicators", tags=["Indicators"])


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
    dates: tuple[date, date] = Depends(DateRange(365)),
    db: Session = Depends(get_db),
):
    """
    Get the time series for a single FRED indicator.

    **Examples:**
    - Federal Funds Rate: `/api/indicators/DFF`
    - CPI last year: `/api/indicators/CPIAUCSL?start_date=2025-03-01`
    - GDP all time: `/api/indicators/GDP`
    """
    start_date, end_date = dates

    return get_indicators_by_series(db, series_id, start_date, end_date)
