"""
Prefect flow for daily analysis computations.

Pre-computes correlations and historical patterns so the
frontend API endpoints return results quickly.
"""

from datetime import date, timedelta
from prefect import flow, task, get_run_logger

from src.analysis.correlation import CorrelationAnalyzer
from src.analysis.historical_patterns import HistoricalPatternAnalyzer
from src.analysis.sentiment import score_unprocessed_headlines
from src.analysis.embeddings import embed_unprocessed_headlines, embed_unprocessed_events
from src.config.constants import get_all_symbols, SYMBOLS


@task(name="compute-correlations", retries=1, retry_delay_seconds=30, log_prints=True)
def compute_correlations() -> dict:
    """Compute top correlation pairs across all symbols and cache results."""
    logger = get_run_logger()
    end_date = date.today()
    start_date = date(2016, 1, 1)

    # Compute across ALL symbols for complete cache
    all_symbols = get_all_symbols()

    logger.info(f"Computing correlations for {len(all_symbols)} symbols ({start_date} to {end_date})")

    analyzer = CorrelationAnalyzer()
    try:
        top_pairs = analyzer.top_correlated_pairs(all_symbols, start_date, end_date, limit=200)
        n_pairs = len(top_pairs) if top_pairs is not None else 0
        logger.info(f"Found {n_pairs} correlation pairs")

        # Save to cache
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


@task(name="score-headline-sentiment", retries=1, retry_delay_seconds=30, log_prints=True)
def score_sentiment() -> dict:
    """Score sentiment on all unprocessed news headlines."""
    logger = get_run_logger()
    logger.info("Scoring sentiment on unprocessed headlines")

    try:
        from src.db.connection import get_session
        with get_session() as session:
            n_scored = score_unprocessed_headlines(session)
        logger.info(f"Scored {n_scored} headlines")
        return {"headlines_scored": n_scored}
    except Exception as e:
        logger.warning(f"Sentiment scoring failed: {e}")
        return {"headlines_scored": 0, "error": str(e)}


@task(name="generate-embeddings", retries=1, retry_delay_seconds=30, log_prints=True)
def generate_embeddings() -> dict:
    """Generate embeddings for unprocessed headlines and events."""
    logger = get_run_logger()
    logger.info("Generating embeddings for unprocessed content")

    try:
        from src.db.connection import get_session
        with get_session() as session:
            n_headlines = embed_unprocessed_headlines(session)
        with get_session() as session:
            n_events = embed_unprocessed_events(session)
        logger.info(f"Embedded {n_headlines} headlines, {n_events} events")
        return {"headlines_embedded": n_headlines, "events_embedded": n_events}
    except Exception as e:
        logger.warning(f"Embedding generation failed: {e}")
        return {"headlines_embedded": 0, "events_embedded": 0, "error": str(e)}


@flow(name="daily-analysis", retries=1, log_prints=True)
def daily_analysis() -> dict:
    """Run daily analysis: sentiment + embeddings + correlations + patterns."""
    logger = get_run_logger()
    logger.info("Starting daily analysis pipeline")

    sentiment_result = score_sentiment()
    embedding_result = generate_embeddings()
    corr_result = compute_correlations()
    pattern_result = compute_patterns()

    logger.info("Daily analysis complete")
    return {
        "sentiment": sentiment_result,
        "embeddings": embedding_result,
        "correlations": corr_result,
        "patterns": pattern_result,
    }


if __name__ == "__main__":
    daily_analysis()
