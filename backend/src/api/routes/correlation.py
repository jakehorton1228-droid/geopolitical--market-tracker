"""
Correlation API Router.

Endpoints for analyzing correlations between geopolitical events and market returns.
"""

from datetime import date
import logging
from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.deps import DateRange
from src.api.schemas import CorrelationResponse, RollingCorrelationResponse, RollingCorrelationPoint
from src.config.constants import get_all_symbols

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/correlation", tags=["Correlation"])


# NOTE: Static path routes (/top, /heatmap) must come BEFORE
# parameterized routes (/{symbol}) so FastAPI doesn't match
# "top" or "heatmap" as a symbol.


@router.get("/top", response_model=list[dict])
def get_top_correlations(
    dates: tuple[date, date] = Depends(DateRange(365)),
    limit: int = Query(20, ge=1, le=100),
    method: str = Query("pearson"),
    symbols: list[str] | None = Query(None, description="Symbols to include; repeat to pass multiple (default: all)"),
):
    """
    Find the strongest correlations across all symbol-event metric pairs.

    Uses pre-computed cache when available (populated by daily Prefect pipeline).
    Falls back to live computation if cache is empty or specific symbols are requested.

    Note: intentionally typed as ``dict`` — results come from either the
    correlation cache or a live DataFrame, whose columns are not a fixed contract.
    """
    start_date, end_date = dates

    # Try cache first (only for default all-symbols requests)
    if not symbols:
        from src.db.connection import get_session
        from src.db.queries import get_cached_correlations

        with get_session() as session:
            cached = get_cached_correlations(session, method=method, limit=limit)
            if cached:
                logger.info(f"Returning {len(cached)} cached correlation results")
                return cached

    # Cache miss or specific symbols requested — compute live
    from src.analysis.correlation import CorrelationAnalyzer

    symbol_list = symbols if symbols else get_all_symbols()

    analyzer = CorrelationAnalyzer()
    df = analyzer.top_correlated_pairs(symbol_list, start_date, end_date, method, limit)

    if df.empty:
        return []

    return df.to_dict(orient="records")


@router.get("/heatmap", response_model=dict)
def get_correlation_heatmap(
    symbols: list[str] = Query(..., description="Symbols to include; repeat to pass multiple (e.g. ?symbols=CL=F&symbols=SPY)"),
    dates: tuple[date, date] = Depends(DateRange(365)),
    method: str = Query("pearson"),
):
    """
    Compute correlation matrix: symbols x event_metrics.

    Returns data suitable for rendering a heatmap.

    Note: intentionally typed as ``dict`` — the payload is a matrix whose
    axes depend on the requested symbols and available metrics.
    """
    from src.analysis.correlation import CorrelationAnalyzer

    start_date, end_date = dates

    analyzer = CorrelationAnalyzer()
    return analyzer.correlation_heatmap(symbols, start_date, end_date, method)


@router.get("/{symbol}", response_model=list[CorrelationResponse])
def get_correlations(
    symbol: str,
    dates: tuple[date, date] = Depends(DateRange(365)),
    method: str = Query("pearson", description="pearson or spearman"),
):
    """
    Get correlation between each event metric and returns for a symbol.

    Returns one result per event metric (goldstein_mean, mentions_total, etc.)
    """
    from src.analysis.correlation import CorrelationAnalyzer

    start_date, end_date = dates

    analyzer = CorrelationAnalyzer()
    results = analyzer.compute_correlations(symbol, start_date, end_date, method)

    if not results:
        raise HTTPException(status_code=404, detail=f"Insufficient data for {symbol}")

    return [
        CorrelationResponse(
            symbol=r.symbol,
            event_metric=r.event_metric,
            correlation=round(r.correlation, 4),
            p_value=round(r.p_value, 4),
            n_observations=r.n_observations,
            method=r.method,
        )
        for r in results
    ]


@router.get("/{symbol}/rolling", response_model=RollingCorrelationResponse)
def get_rolling_correlation(
    symbol: str,
    dates: tuple[date, date] = Depends(DateRange(365)),
    event_metric: str = Query("conflict_count", description="Event metric to correlate"),
    window_days: int = Query(30, ge=10, le=180),
):
    """
    Get rolling window correlation between an event metric and returns.

    Returns a timeseries of correlation values with confidence intervals.
    """
    from src.analysis.correlation import CorrelationAnalyzer

    start_date, end_date = dates

    analyzer = CorrelationAnalyzer()
    result = analyzer.compute_rolling_correlation(
        symbol, start_date, end_date, event_metric, window_days
    )

    if result is None:
        raise HTTPException(status_code=404, detail="Insufficient data for rolling correlation")

    return RollingCorrelationResponse(
        symbol=symbol,
        event_metric=result.event_metric,
        window_days=result.window_days,
        data=[
            RollingCorrelationPoint(
                date=str(d),
                correlation=round(c, 4),
                upper_ci=round(u, 4),
                lower_ci=round(l, 4),
            )
            for d, c, u, l in zip(
                result.dates, result.correlations,
                result.upper_ci, result.lower_ci
            )
        ],
    )
