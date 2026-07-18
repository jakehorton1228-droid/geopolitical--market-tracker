"""Prefect flow for the analytics stage — statistical inference in Python.

Fourth/last stage of the daily pipeline:  ingest → enrich → transform → analytics.

This is the statistical-inference work that isn't SQL and therefore isn't in the
dbt transform layer: correlations (Pearson r + p-values) and historical patterns.
It runs LAST because it reads the freshly-built Silver/Gold layer (e.g. correlations
read silver_market via feature_engineering).

Results are cached in Postgres (correlation_cache, etc.) so the API serves them fast.

Note: sentiment scoring and embeddings moved to enrich_flow.py (they run before the
transform, since silver_headlines depends on sentiment).
"""

from datetime import date
from prefect import flow, task, get_run_logger

from src.analysis.correlation import CorrelationAnalyzer
from src.analysis.historical_patterns import HistoricalPatternAnalyzer
from src.config.constants import get_all_symbols


@task(name="compute-correlations", retries=1, retry_delay_seconds=30, log_prints=True)
def compute_correlations() -> dict:
    """Compute top correlation pairs across all symbols and cache results."""
    logger = get_run_logger()
    end_date = date.today()
    start_date = date(2016, 1, 1)

    all_symbols = get_all_symbols()
    logger.info(f"Computing correlations for {len(all_symbols)} symbols ({start_date} to {end_date})")

    analyzer = CorrelationAnalyzer()
    try:
        top_pairs = analyzer.top_correlated_pairs(all_symbols, start_date, end_date, limit=200)
        n_pairs = len(top_pairs) if top_pairs is not None else 0
        logger.info(f"Found {n_pairs} correlation pairs")

        if not top_pairs.empty:
            from src.db.connection import get_session
            from src.db.queries import save_correlation_cache

            records = top_pairs.to_dict(orient="records")
            for r in records:
                r["start_date"] = start_date
                r["end_date"] = end_date

            with get_session() as session:
                saved = save_correlation_cache(session, records, method="pearson")
                logger.info(f"Cached {saved} correlation results to database")

        return {"pairs_computed": n_pairs}
    except Exception as e:
        logger.warning(f"Correlation computation failed: {e}")
        return {"pairs_computed": 0, "error": str(e)}


@task(name="compute-patterns", retries=1, retry_delay_seconds=30, log_prints=True)
def compute_patterns() -> dict:
    """Compute historical patterns for key symbols."""
    logger = get_run_logger()
    end_date = date.today()
    start_date = date(2016, 1, 1)

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
    """Analytics stage: correlations + historical patterns over the Silver/Gold layer."""
    logger = get_run_logger()
    logger.info("=== Starting Analytics ===")

    corr_result = compute_correlations()
    pattern_result = compute_patterns()

    logger.info("=== Analytics Complete ===")
    return {
        "correlations": corr_result,
        "patterns": pattern_result,
    }


if __name__ == "__main__":
    daily_analysis()
