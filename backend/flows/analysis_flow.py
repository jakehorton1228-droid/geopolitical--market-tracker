"""
Prefect flow for daily analysis computations.

Pre-computes correlations and historical patterns so the
frontend API endpoints return results quickly.
"""

from datetime import date, timedelta
from prefect import flow, task, get_run_logger

from src.analysis.correlation import CorrelationAnalyzer
from src.analysis.historical_patterns import HistoricalPatternAnalyzer
from src.config.constants import get_all_symbols, SYMBOLS


@task(name="compute-correlations", retries=1, retry_delay_seconds=30, log_prints=True)
def compute_correlations(days_back: int = 365) -> dict:
    """Compute top correlation pairs across all symbols."""
    logger = get_run_logger()
    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)

    # Use a focused subset for daily computation (commodities + key ETFs)
    key_symbols = (
        SYMBOLS.get("commodities", []) +
        SYMBOLS.get("volatility", []) +
        ["SPY", "QQQ", "EEM", "TLT"]
    )

    logger.info(f"Computing correlations for {len(key_symbols)} symbols ({start_date} to {end_date})")

    analyzer = CorrelationAnalyzer()
    try:
        top_pairs = analyzer.top_correlated_pairs(key_symbols, start_date, end_date, limit=30)
        n_pairs = len(top_pairs) if top_pairs is not None else 0
        logger.info(f"Found {n_pairs} significant correlation pairs")
        return {"pairs_computed": n_pairs}
    except Exception as e:
        logger.warning(f"Correlation computation failed: {e}")
        return {"pairs_computed": 0, "error": str(e)}


@task(name="compute-patterns", retries=1, retry_delay_seconds=30, log_prints=True)
def compute_patterns(days_back: int = 365) -> dict:
    """Compute historical patterns for key symbols."""
    logger = get_run_logger()
    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)

    # Focus on most-traded symbols
    key_symbols = ["CL=F", "GC=F", "SPY", "QQQ", "^VIX", "EURUSD=X", "TLT", "EEM"]

    logger.info(f"Computing patterns for {len(key_symbols)} symbols ({start_date} to {end_date})")

    analyzer = HistoricalPatternAnalyzer()
    total_patterns = 0

    for symbol in key_symbols:
        try:
            patterns = analyzer.all_patterns_for_symbol(symbol, start_date, end_date, min_occurrences=5)
            n = len(patterns) if patterns else 0
            total_patterns += n
            logger.info(f"  {symbol}: {n} patterns")
        except Exception as e:
            logger.warning(f"  {symbol}: failed - {e}")

    logger.info(f"Computed {total_patterns} total patterns")
    return {"total_patterns": total_patterns, "symbols_processed": len(key_symbols)}


@flow(name="daily-analysis", retries=1, log_prints=True)
def daily_analysis() -> dict:
    """Run daily analysis: correlations + patterns."""
    logger = get_run_logger()
    logger.info("Starting daily analysis pipeline")

    corr_result = compute_correlations()
    pattern_result = compute_patterns()

    logger.info("Daily analysis complete")
    return {"correlations": corr_result, "patterns": pattern_result}


if __name__ == "__main__":
    daily_analysis()
