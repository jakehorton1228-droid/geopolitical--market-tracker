"""
Prefect flow for daily data ingestion.

Ingests GDELT geopolitical events, Yahoo Finance market data,
RSS news headlines, and FRED economic indicators.

Uses a 3-day overlap for events (late-arriving data) and 7-day
window for market data (covers weekends/holidays). RSS feeds
are fetched in full each run — deduplication happens at the DB level.
FRED series are fetched with a 90-day window to catch revisions and
backfill any gaps.

This flow is pure extract-load: it lands raw data in the Bronze tables and does
nothing else. Enrichment (sentiment + embeddings) is the next pipeline stage
(enrich_flow.py); transforms run after that (dbt).
"""

from datetime import date, timedelta
from prefect import flow, task, get_run_logger

from src.ingestion.gdelt import GDELTIngestion
from src.ingestion.market_data import MarketDataIngestion
from src.ingestion.rss_feeds import RSSIngestion
from src.ingestion.fred import FREDIngestion
from src.ingestion.polymarket import PolymarketIngestion


@task(name="ingest-events", retries=3, retry_delay_seconds=60, log_prints=True)
def ingest_events(days_back: int = 3) -> dict:
    """Ingest GDELT events for the last N days."""
    logger = get_run_logger()
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=days_back - 1)

    logger.info(f"Ingesting GDELT events from {start_date} to {end_date}")

    g = GDELTIngestion()
    result = g.ingest_date_range(start_date, end_date)

    total = sum(result.values()) if isinstance(result, dict) else result
    logger.info(f"Ingested {total} events across {days_back} days")

    return {"events_ingested": total, "start_date": str(start_date), "end_date": str(end_date)}


@task(name="ingest-market", retries=3, retry_delay_seconds=60, log_prints=True)
def ingest_market(days_back: int = 7) -> dict:
    """Ingest market data for all tracked symbols."""
    logger = get_run_logger()
    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)

    logger.info(f"Ingesting market data from {start_date} to {end_date}")

    m = MarketDataIngestion()
    result = m.ingest_all_symbols(start_date, end_date)

    total = sum(v for v in result.values() if v > 0) if isinstance(result, dict) else 0
    symbols_ok = sum(1 for v in result.values() if v > 0) if isinstance(result, dict) else 0
    logger.info(f"Ingested {total} rows for {symbols_ok} symbols")

    return {"rows_ingested": total, "symbols_updated": symbols_ok}


@task(name="ingest-rss", retries=2, retry_delay_seconds=60, log_prints=True)
def ingest_rss() -> dict:
    """Ingest news headlines from all configured RSS feeds."""
    logger = get_run_logger()
    logger.info("Ingesting RSS news headlines")

    rss = RSSIngestion()
    result = rss.ingest_all_feeds()

    total = sum(v for v in result.values() if v > 0)
    feeds_ok = sum(1 for v in result.values() if v >= 0)
    logger.info(f"Ingested {total} new headlines from {feeds_ok} feeds")

    return {"headlines_ingested": total, "feeds": result}


@task(name="ingest-fred", retries=2, retry_delay_seconds=60, log_prints=True)
def ingest_fred(days_back: int = 90) -> dict:
    """
    Ingest economic indicators from FRED.

    Uses a 90-day window by default because:
    - Monthly series (CPI, unemployment) only update once/month
    - Quarterly series (GDP) only update once/quarter
    - A wide window ensures we catch new releases without tracking release dates
    - Deduplication at the DB level means overlap is free
    """
    logger = get_run_logger()
    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)

    logger.info(f"Ingesting FRED indicators from {start_date} to {end_date}")

    fred = FREDIngestion()
    result = fred.ingest_all_series(start_date=start_date, end_date=end_date)

    total = sum(v for v in result.values() if v > 0)
    series_ok = sum(1 for v in result.values() if v >= 0)
    logger.info(f"Ingested {total} new observations from {series_ok} series")

    return {"observations_ingested": total, "series": result}


@task(name="ingest-polymarket", retries=2, retry_delay_seconds=60, log_prints=True)
def ingest_polymarket() -> dict:
    """
    Ingest geopolitical prediction market odds from Polymarket.

    Stores one daily snapshot per market. Probabilities are forward-looking
    risk signals — when "US-Iran ceasefire" odds drop from 40% to 20%,
    that's a signal worth tracking before financial markets react.
    """
    logger = get_run_logger()
    logger.info("Ingesting Polymarket prediction markets")

    pm = PolymarketIngestion()
    count = pm.ingest_markets()

    logger.info(f"Ingested {count} prediction market snapshots")
    return {"snapshots_ingested": count}


@flow(name="daily-ingestion", log_prints=True)
def daily_ingestion() -> dict:
    """Run daily extract-load: events + market data + RSS + FRED + Polymarket.

    Pure extract-load — lands raw data in Bronze, nothing more. Enrichment
    (sentiment + embeddings) and transforms are separate downstream stages.

    Uses a "continue on failure" pattern — each source is independent, so one
    source failing (e.g., missing API key, API downtime) should NOT block the
    others. This is a core data engineering principle: pipelines should be
    resilient. If FRED is down, GDELT data should still flow through.

    Each task is wrapped in try/except. Failures are logged and tracked in the
    results dict, but the flow continues. The return value tells you exactly
    what succeeded and what didn't.
    """
    logger = get_run_logger()
    logger.info("Starting daily ingestion pipeline")

    results = {}
    failures = []

    # Each source runs independently — one failure doesn't block others
    for name, task_fn in [
        ("events", ingest_events),
        ("market", ingest_market),
        ("rss", ingest_rss),
        ("fred", ingest_fred),
        ("polymarket", ingest_polymarket),
    ]:
        try:
            results[name] = task_fn()
        except Exception as e:
            logger.warning(f"{name} ingestion failed: {e}")
            results[name] = {"status": "failed", "error": str(e)}
            failures.append(name)

    if failures:
        logger.warning(f"Ingestion completed with failures: {failures}")
    else:
        logger.info("All ingestion tasks succeeded")

    logger.info("Daily ingestion complete")
    return results


if __name__ == "__main__":
    daily_ingestion()
