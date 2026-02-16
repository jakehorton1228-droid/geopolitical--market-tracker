"""
Correlation API Router.

Endpoints for analyzing correlations between geopolitical events and market returns.
"""

from datetime import date, timedelta
import logging
from fastapi import APIRouter, HTTPException, Query

from src.config.constants import get_all_symbols

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/correlation", tags=["Correlation"])


# NOTE: Static path routes (/top, /heatmap) must come BEFORE
# parameterized routes (/{symbol}) so FastAPI doesn't match
# "top" or "heatmap" as a symbol.


@router.get("/top", response_model=list[dict])
def get_top_correlations(
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    method: str = Query("pearson"),
    symbols: str | None = Query(None, description="Comma-separated symbols (default: all)"),
):
    """
    Find the strongest correlations across all symbol-event metric pairs.

    Powers the dashboard home "Top Correlated Pairs" widget.
    """
    from src.analysis.correlation import CorrelationAnalyzer

    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=365)

    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]
    else:
        symbol_list = get_all_symbols()

    analyzer = CorrelationAnalyzer()
    df = analyzer.top_correlated_pairs(symbol_list, start_date, end_date, method, limit)

    if df.empty:
        return []

    return df.to_dict(orient="records")


@router.get("/heatmap", response_model=dict)
def get_correlation_heatmap(
    symbols: str = Query(..., description="Comma-separated symbols"),
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    method: str = Query("pearson"),
):
    """
    Compute correlation matrix: symbols x event_metrics.

    Returns data suitable for rendering a heatmap.
    """
    from src.analysis.correlation import CorrelationAnalyzer

    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=365)

    symbol_list = [s.strip() for s in symbols.split(",")]

    analyzer = CorrelationAnalyzer()
    return analyzer.correlation_heatmap(symbol_list, start_date, end_date, method)


@router.get("/{symbol}", response_model=list[dict])
def get_correlations(
    symbol: str,
    start_date: date | None = Query(None, description="Start date (default: 365 days ago)"),
    end_date: date | None = Query(None, description="End date (default: today)"),
    method: str = Query("pearson", description="pearson or spearman"),
):
    """
    Get correlation between each event metric and returns for a symbol.

    Returns one result per event metric (goldstein_mean, mentions_total, etc.)
    """
    from src.analysis.correlation import CorrelationAnalyzer

    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=365)

    analyzer = CorrelationAnalyzer()
    results = analyzer.compute_correlations(symbol, start_date, end_date, method)

    if not results:
        raise HTTPException(status_code=422, detail=f"Insufficient data for {symbol}")

    return [
        {
            "symbol": r.symbol,
            "event_metric": r.event_metric,
            "correlation": round(r.correlation, 4),
            "p_value": round(r.p_value, 4),
            "n_observations": r.n_observations,
            "method": r.method,
        }
        for r in results
    ]


@router.get("/{symbol}/rolling", response_model=dict)
def get_rolling_correlation(
    symbol: str,
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    event_metric: str = Query("conflict_count", description="Event metric to correlate"),
    window_days: int = Query(30, ge=10, le=180),
):
    """
    Get rolling window correlation between an event metric and returns.

    Returns a timeseries of correlation values with confidence intervals.
    """
    from src.analysis.correlation import CorrelationAnalyzer

    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=365)

    analyzer = CorrelationAnalyzer()
    result = analyzer.compute_rolling_correlation(
        symbol, start_date, end_date, event_metric, window_days
    )

    if result is None:
        raise HTTPException(status_code=422, detail=f"Insufficient data for rolling correlation")

    return {
        "symbol": symbol,
        "event_metric": result.event_metric,
        "window_days": result.window_days,
        "data": [
            {
                "date": str(d),
                "correlation": round(c, 4),
                "upper_ci": round(u, 4),
                "lower_ci": round(l, 4),
            }
            for d, c, u, l in zip(
                result.dates, result.correlations,
                result.upper_ci, result.lower_ci
            )
        ],
    }
