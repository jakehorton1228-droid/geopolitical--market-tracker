"""
GDELT Event Ingestion Flow.

Fetches geopolitical events from GDELT and stores them in the database.

USAGE:
------
    # Run directly
    python flows/ingest_events.py

    # Run via Prefect CLI
    prefect deployment run 'ingest-gdelt-events/daily'

    # Import and run programmatically
    from flows.ingest_events import ingest_gdelt_events
    ingest_gdelt_events(days_back=7)

CONCEPTS:
---------
    @task: A single unit of work (fetch data, transform, load)
    @flow: Orchestrates multiple tasks, handles retries and logging

    Tasks are like functions with superpowers:
    - Automatic retries on failure
    - Caching to avoid re-running
    - Logging and observability
    - Concurrent execution
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import date, timedelta
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from datetime import timedelta as td

from src.ingestion.gdelt import GDELTIngestion
from src.db.connection import get_session
from src.db.models import Event


@task(
    name="fetch-gdelt-daily",
    description="Fetch GDELT events for a single day",
    retries=3,
    retry_delay_seconds=60,
    cache_key_fn=task_input_hash,
    cache_expiration=td(hours=12),
)
def fetch_gdelt_for_date(target_date: date) -> dict:
    """
    Fetch GDELT events for a specific date.

    This task:
    1. Downloads the daily GDELT CSV file
    2. Filters for significant events (5+ mentions)
    3. Returns a summary of what was fetched

    The @task decorator adds:
    - retries=3: If it fails, try up to 3 more times
    - retry_delay_seconds=60: Wait 1 minute between retries
    - cache_key_fn: Cache results based on input (date)
    - cache_expiration: Cache is valid for 12 hours
    """
    logger = get_run_logger()
    logger.info(f"Fetching GDELT events for {target_date}")

    ingestion = GDELTIngestion()

    try:
        # ingest_date fetches and stores in one call
        event_count = ingestion.ingest_date(target_date)

        if event_count > 0:
            logger.info(f"Stored {event_count} events for {target_date}")
            return {
                "date": str(target_date),
                "status": "success",
                "event_count": event_count,
            }
        else:
            logger.warning(f"No events found for {target_date}")
            return {
                "date": str(target_date),
                "status": "no_data",
                "event_count": 0,
            }

    except Exception as e:
        logger.error(f"Failed to fetch events for {target_date}: {e}")
        raise  # Re-raise so Prefect knows to retry


@task(name="get-existing-dates", description="Check which dates already have data")
def get_existing_event_dates(start_date: date, end_date: date) -> set[date]:
    """
    Check which dates already have events in the database.

    This helps us avoid re-fetching data we already have,
    making the pipeline idempotent (safe to run multiple times).
    """
    logger = get_run_logger()

    with get_session() as session:
        from sqlalchemy import func

        existing = session.query(
            func.distinct(Event.event_date)
        ).filter(
            Event.event_date >= start_date,
            Event.event_date <= end_date,
        ).all()

        existing_dates = {row[0] for row in existing}
        logger.info(f"Found {len(existing_dates)} dates with existing data")
        return existing_dates


@task(name="summarize-ingestion", description="Create summary of ingestion results")
def summarize_results(results: list[dict]) -> dict:
    """Summarize the ingestion results."""
    logger = get_run_logger()

    total_events = sum(r.get("event_count", 0) for r in results)
    successful = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "failed")
    no_data = sum(1 for r in results if r.get("status") == "no_data")

    summary = {
        "total_dates_processed": len(results),
        "successful": successful,
        "failed": failed,
        "no_data": no_data,
        "total_events_ingested": total_events,
    }

    logger.info(f"Ingestion complete: {summary}")
    return summary


@flow(
    name="ingest-gdelt-events",
    description="Fetch and store GDELT geopolitical events",
    version="1.0.0",
)
def ingest_gdelt_events(
    days_back: int = 7,
    end_date: date | None = None,
    skip_existing: bool = True,
) -> dict:
    """
    Main flow to ingest GDELT events.

    This flow orchestrates the entire ingestion process:
    1. Determine date range to fetch
    2. Check which dates already have data (optional)
    3. Fetch events for each date (with retries)
    4. Summarize results

    Parameters
    ----------
    days_back : int
        Number of days to look back from end_date (default: 7)
    end_date : date, optional
        End date for ingestion (default: yesterday)
    skip_existing : bool
        Skip dates that already have data (default: True)

    Returns
    -------
    dict
        Summary of ingestion results

    Example
    -------
    >>> # Ingest last 7 days
    >>> ingest_gdelt_events(days_back=7)

    >>> # Ingest specific range
    >>> ingest_gdelt_events(days_back=30, end_date=date(2024, 1, 15))
    """
    logger = get_run_logger()

    # Determine date range
    if end_date is None:
        end_date = date.today() - timedelta(days=1)  # Yesterday

    start_date = end_date - timedelta(days=days_back - 1)

    logger.info(f"Starting GDELT ingestion: {start_date} to {end_date}")

    # Generate list of dates to process
    dates_to_process = [
        start_date + timedelta(days=i)
        for i in range(days_back)
    ]

    # Optionally skip dates with existing data
    if skip_existing:
        existing_dates = get_existing_event_dates(start_date, end_date)
        dates_to_process = [d for d in dates_to_process if d not in existing_dates]
        logger.info(f"Skipping {len(existing_dates)} dates with existing data")

    if not dates_to_process:
        logger.info("No new dates to process")
        return {"status": "skipped", "message": "All dates already have data"}

    logger.info(f"Processing {len(dates_to_process)} dates")

    # Fetch events for each date
    # Note: These run sequentially by default. For parallel execution,
    # you could use task.map() or task.submit()
    results = []
    for target_date in dates_to_process:
        try:
            result = fetch_gdelt_for_date(target_date)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {target_date}: {e}")
            results.append({
                "date": str(target_date),
                "status": "failed",
                "error": str(e),
            })

    # Summarize results
    summary = summarize_results(results)

    return summary


# ============================================================================
# DEPLOYMENT CONFIGURATION
# ============================================================================
# Deployments define HOW and WHEN a flow runs.
# You can create deployments via CLI or Python.

def create_deployment():
    """
    Create a Prefect deployment for scheduled runs.

    Run this once to register the deployment:
        python flows/ingest_events.py --deploy

    Then the flow will run automatically on schedule.
    """
    from prefect.deployments import Deployment
    from prefect.server.schemas.schedules import CronSchedule

    deployment = Deployment.build_from_flow(
        flow=ingest_gdelt_events,
        name="daily",
        version="1.0.0",
        description="Daily GDELT event ingestion - runs at 6 AM UTC",
        schedule=CronSchedule(cron="0 6 * * *"),  # 6 AM UTC daily
        parameters={"days_back": 1, "skip_existing": True},
        tags=["ingestion", "gdelt", "daily"],
    )

    deployment.apply()
    print("Deployment created: ingest-gdelt-events/daily")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GDELT Event Ingestion")
    parser.add_argument("--deploy", action="store_true", help="Create Prefect deployment")
    parser.add_argument("--days", type=int, default=7, help="Days to look back")

    args = parser.parse_args()

    if args.deploy:
        create_deployment()
    else:
        # Run the flow directly
        result = ingest_gdelt_events(days_back=args.days)
        print(f"Result: {result}")
