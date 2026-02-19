"""
Historical Patterns API Router.

Endpoints for querying historical frequency patterns.
"When X happens, Y goes UP Z% of the time."
"""

from datetime import date, timedelta
import logging
from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/patterns", tags=["Historical Patterns"])


@router.get("/{symbol}", response_model=dict)
def get_pattern(
    symbol: str,
    start_date: date | None = Query(None, description="Start date (default: 365 days ago)"),
    end_date: date | None = Query(None, description="End date (default: today)"),
    event_group: str | None = Query(None, description="Event group: violent_conflict, material_conflict, etc."),
    country_code: str | None = Query(None, description="3-letter ISO country code"),
    min_event_count: int = Query(1, ge=1, description="Minimum events per day to count"),
):
    """
    Get historical frequency pattern for a symbol given event filters.

    Returns: up/down counts, percentages, average returns, and statistical significance.
    """
    from src.analysis.historical_patterns import HistoricalPatternAnalyzer

    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = date(2016, 1, 1)

    analyzer = HistoricalPatternAnalyzer()
    pattern = analyzer.analyze_event_type_pattern(
        symbol, start_date, end_date,
        event_group=event_group,
        country_code=country_code,
        min_event_count=min_event_count,
    )

    if pattern is None:
        raise HTTPException(
            status_code=422,
            detail=f"Insufficient data for pattern analysis on {symbol}",
        )

    return {
        "symbol": pattern.symbol,
        "event_filter": pattern.event_filter,
        "total_occurrences": pattern.total_occurrences,
        "up_count": pattern.up_count,
        "down_count": pattern.down_count,
        "up_percentage": round(pattern.up_percentage, 1),
        "avg_return_up": round(pattern.avg_return_up, 3),
        "avg_return_down": round(pattern.avg_return_down, 3),
        "avg_return_all": round(pattern.avg_return_all, 3),
        "median_return": round(pattern.median_return, 3),
        "t_statistic": round(pattern.t_statistic, 3),
        "p_value": round(pattern.p_value, 4),
        "is_significant": pattern.p_value < 0.05,
    }


@router.get("/{symbol}/all", response_model=list[dict])
def get_all_patterns(
    symbol: str,
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    min_occurrences: int = Query(10, ge=5, description="Minimum occurrences to include"),
):
    """
    Get all historical patterns for a symbol.

    Returns patterns for all event groups and relevant countries,
    sorted by statistical significance.
    """
    from src.analysis.historical_patterns import HistoricalPatternAnalyzer

    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = date(2016, 1, 1)

    analyzer = HistoricalPatternAnalyzer()
    patterns = analyzer.all_patterns_for_symbol(
        symbol, start_date, end_date, min_occurrences
    )

    return [
        {
            "symbol": p.symbol,
            "event_filter": p.event_filter,
            "total_occurrences": p.total_occurrences,
            "up_count": p.up_count,
            "down_count": p.down_count,
            "up_percentage": round(p.up_percentage, 1),
            "avg_return_up": round(p.avg_return_up, 3),
            "avg_return_down": round(p.avg_return_down, 3),
            "avg_return_all": round(p.avg_return_all, 3),
            "median_return": round(p.median_return, 3),
            "t_statistic": round(p.t_statistic, 3),
            "p_value": round(p.p_value, 4),
            "is_significant": p.p_value < 0.05,
        }
        for p in patterns
    ]
