"""
Prefect flow for daily data ingestion.

Ingests GDELT geopolitical events and Yahoo Finance market data.
Uses a 3-day overlap for events (late-arriving data) and 7-day
window for market data (covers weekends/holidays).
"""

from datetime import date, timedelta
from prefect import flow, task, get_run_logger

from src.ingestion.gdelt import GDELTIngestion
from src.ingestion.market_data import MarketDataIngestion


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


@flow(name="daily-ingestion", retries=2, retry_delay_seconds=300, log_prints=True)
def daily_ingestion() -> dict:
    """Run daily data ingestion: events + market data."""
    logger = get_run_logger()
    logger.info("Starting daily ingestion pipeline")

    events_result = ingest_events()
    market_result = ingest_market()

    logger.info("Daily ingestion complete")
    return {"events": events_result, "market": market_result}


if __name__ == "__main__":
    daily_ingestion()
