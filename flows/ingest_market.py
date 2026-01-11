"""
Market Data Ingestion Flow.

Fetches OHLCV data from Yahoo Finance and stores it in the database.

USAGE:
------
    # Run directly
    python flows/ingest_market.py

    # Run via Prefect CLI
    prefect deployment run 'ingest-market-data/daily'

    # Import and run programmatically
    from flows.ingest_market import ingest_market_data
    ingest_market_data(days_back=30)

CONCEPTS:
---------
    Parallel Tasks: Using .map() to process multiple symbols concurrently
    Batching: Processing symbols in groups to avoid rate limits
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import date, timedelta
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from datetime import timedelta as td
import time

from src.ingestion.market_data import MarketDataIngestion
from src.config.constants import TRACKED_SYMBOLS
from src.db.connection import get_session
from src.db.models import MarketData


@task(
    name="fetch-symbol-data",
    description="Fetch market data for a single symbol",
    retries=2,
    retry_delay_seconds=30,
    cache_key_fn=task_input_hash,
    cache_expiration=td(hours=6),
)
def fetch_symbol_data(
    symbol: str,
    start_date: date,
    end_date: date,
) -> dict:
    """
    Fetch market data for a single symbol.

    Yahoo Finance has rate limits, so we:
    - Retry on failure
    - Cache results to avoid re-fetching
    """
    logger = get_run_logger()
    logger.info(f"Fetching {symbol} from {start_date} to {end_date}")

    ingestion = MarketDataIngestion()

    try:
        # ingest_symbol fetches and stores in one call
        record_count = ingestion.ingest_symbol(symbol, start_date, end_date)

        if record_count > 0:
            logger.info(f"Stored {record_count} records for {symbol}")
            return {
                "symbol": symbol,
                "status": "success",
                "record_count": record_count,
            }
        else:
            logger.warning(f"No data returned for {symbol}")
            return {
                "symbol": symbol,
                "status": "no_data",
                "record_count": 0,
            }

    except Exception as e:
        logger.error(f"Failed to fetch {symbol}: {e}")
        raise


@task(name="get-symbols-needing-update", description="Find symbols with stale data")
def get_symbols_needing_update(
    symbols: list[str],
    end_date: date,
    max_staleness_days: int = 2,
) -> list[str]:
    """
    Find symbols that need updating.

    A symbol needs updating if:
    - It has no data at all
    - Its most recent data is older than max_staleness_days
    """
    logger = get_run_logger()

    stale_threshold = end_date - timedelta(days=max_staleness_days)

    with get_session() as session:
        from sqlalchemy import func

        # Get most recent date for each symbol
        latest_dates = session.query(
            MarketData.symbol,
            func.max(MarketData.date).label("latest_date"),
        ).filter(
            MarketData.symbol.in_(symbols),
        ).group_by(
            MarketData.symbol,
        ).all()

        latest_by_symbol = {row.symbol: row.latest_date for row in latest_dates}

    # Find symbols that need updating
    symbols_to_update = []
    for symbol in symbols:
        latest = latest_by_symbol.get(symbol)
        if latest is None or latest < stale_threshold:
            symbols_to_update.append(symbol)
            logger.debug(f"{symbol}: needs update (latest: {latest})")

    logger.info(f"{len(symbols_to_update)}/{len(symbols)} symbols need updating")
    return symbols_to_update


@task(name="batch-fetch-symbols", description="Fetch multiple symbols with rate limiting")
def batch_fetch_symbols(
    symbols: list[str],
    start_date: date,
    end_date: date,
    delay_seconds: float = 0.5,
) -> list[dict]:
    """
    Fetch multiple symbols with rate limiting.

    Yahoo Finance can rate limit aggressive requests,
    so we add a small delay between symbols.
    """
    logger = get_run_logger()
    results = []

    for i, symbol in enumerate(symbols):
        try:
            result = fetch_symbol_data(symbol, start_date, end_date)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed {symbol}: {e}")
            results.append({
                "symbol": symbol,
                "status": "failed",
                "error": str(e),
            })

        # Rate limiting - sleep between requests (except last one)
        if i < len(symbols) - 1:
            time.sleep(delay_seconds)

    return results


@flow(
    name="ingest-market-data",
    description="Fetch and store market data from Yahoo Finance",
    version="1.0.0",
)
def ingest_market_data(
    days_back: int = 30,
    end_date: date | None = None,
    symbols: list[str] | None = None,
    skip_fresh: bool = True,
) -> dict:
    """
    Main flow to ingest market data.

    Parameters
    ----------
    days_back : int
        Number of days to look back (default: 30)
    end_date : date, optional
        End date for data (default: today)
    symbols : list[str], optional
        Specific symbols to fetch (default: all tracked symbols)
    skip_fresh : bool
        Skip symbols with recent data (default: True)

    Returns
    -------
    dict
        Summary of ingestion results
    """
    logger = get_run_logger()

    # Defaults
    if end_date is None:
        end_date = date.today()

    if symbols is None:
        symbols = TRACKED_SYMBOLS

    start_date = end_date - timedelta(days=days_back)

    logger.info(f"Starting market data ingestion: {start_date} to {end_date}")
    logger.info(f"Symbols to process: {len(symbols)}")

    # Optionally filter to only stale symbols
    if skip_fresh:
        symbols = get_symbols_needing_update(symbols, end_date)
        if not symbols:
            logger.info("All symbols are up to date")
            return {"status": "skipped", "message": "All symbols have fresh data"}

    # Fetch in batches to avoid overwhelming Yahoo Finance
    results = batch_fetch_symbols(symbols, start_date, end_date)

    # Summarize
    total_records = sum(r.get("record_count", 0) for r in results)
    successful = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "failed")

    summary = {
        "total_symbols": len(symbols),
        "successful": successful,
        "failed": failed,
        "total_records_ingested": total_records,
        "date_range": f"{start_date} to {end_date}",
    }

    logger.info(f"Market data ingestion complete: {summary}")
    return summary


def create_deployment():
    """Create Prefect deployment for scheduled runs."""
    from prefect.deployments import Deployment
    from prefect.server.schemas.schedules import CronSchedule

    deployment = Deployment.build_from_flow(
        flow=ingest_market_data,
        name="daily",
        version="1.0.0",
        description="Daily market data ingestion - runs at 7 PM UTC (after market close)",
        schedule=CronSchedule(cron="0 19 * * 1-5"),  # 7 PM UTC, Mon-Fri
        parameters={"days_back": 5, "skip_fresh": True},
        tags=["ingestion", "market", "daily"],
    )

    deployment.apply()
    print("Deployment created: ingest-market-data/daily")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Market Data Ingestion")
    parser.add_argument("--deploy", action="store_true", help="Create Prefect deployment")
    parser.add_argument("--days", type=int, default=30, help="Days to look back")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to fetch")

    args = parser.parse_args()

    if args.deploy:
        create_deployment()
    else:
        result = ingest_market_data(
            days_back=args.days,
            symbols=args.symbols,
        )
        print(f"Result: {result}")
