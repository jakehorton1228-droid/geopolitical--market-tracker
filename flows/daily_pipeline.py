"""
Daily Pipeline - Main Orchestration Flow.

Combines all data ingestion and analysis into a single daily pipeline.

USAGE:
------
    # Run the full pipeline
    python flows/daily_pipeline.py

    # Run via Prefect CLI
    prefect deployment run 'daily-pipeline/production'

    # Start Prefect server (for UI and scheduling)
    prefect server start

CONCEPTS:
---------
    Subflows: The daily pipeline calls other flows as subflows.
    This provides:
    - Modularity: Each flow can run independently
    - Visibility: See progress of each stage in the UI
    - Retries: If one subflow fails, others still complete
    - Reusability: Same flows used in different contexts

PIPELINE STAGES:
----------------
    1. Ingest GDELT events (yesterday's data)
    2. Ingest market data (last 5 trading days)
    3. Run analysis (event studies, anomaly detection)
    4. Generate report

SCHEDULE:
---------
    Runs daily at 8 PM UTC (after US markets close)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import date, timedelta
from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact

# Import subflows
from flows.ingest_events import ingest_gdelt_events
from flows.ingest_market import ingest_market_data
from flows.run_analysis import run_analysis


@task(name="check-prerequisites", description="Verify system is ready to run")
def check_prerequisites() -> dict:
    """
    Check that the system is ready to run the pipeline.

    Verifies:
    - Database connection works
    - Required tables exist
    """
    logger = get_run_logger()

    from src.db.connection import get_session
    from sqlalchemy import text

    try:
        with get_session() as session:
            # Check connection
            session.execute(text("SELECT 1"))

            # Check tables exist
            result = session.execute(text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
            """))
            tables = [row[0] for row in result]

            required_tables = ["events", "market_data", "analysis_results"]
            missing = [t for t in required_tables if t not in tables]

            if missing:
                logger.warning(f"Missing tables: {missing}")
                return {
                    "status": "warning",
                    "message": f"Missing tables: {missing}",
                    "tables": tables,
                }

        logger.info("Prerequisites check passed")
        return {"status": "ok", "tables": tables}

    except Exception as e:
        logger.error(f"Prerequisites check failed: {e}")
        return {"status": "error", "message": str(e)}


@task(name="generate-pipeline-report", description="Create summary report")
def generate_pipeline_report(
    event_results: dict,
    market_results: dict,
    analysis_results: dict,
) -> str:
    """Generate a markdown report of the pipeline run."""
    logger = get_run_logger()

    report = f"""
# Daily Pipeline Report

**Run Date:** {date.today()}

---

## 1. GDELT Event Ingestion

| Metric | Value |
|--------|-------|
| Status | {event_results.get('status', 'unknown')} |
| Dates Processed | {event_results.get('total_dates_processed', 0)} |
| Events Ingested | {event_results.get('total_events_ingested', 0)} |

---

## 2. Market Data Ingestion

| Metric | Value |
|--------|-------|
| Status | {market_results.get('status', 'unknown') if 'status' in market_results else 'success'} |
| Symbols Updated | {market_results.get('successful', 0)} |
| Records Ingested | {market_results.get('total_records_ingested', 0)} |

---

## 3. Analysis

| Metric | Value |
|--------|-------|
| Events Analyzed | {analysis_results.get('events_analyzed', 0)} |
| Event Studies | {analysis_results.get('event_studies', 0)} |
| Significant Results | {analysis_results.get('significant_results', 0)} |
| Anomalies Detected | {analysis_results.get('anomalies_detected', 0)} |

---

## Summary

Pipeline completed successfully. Data is ready for dashboard viewing.

- **Dashboard:** http://localhost:8501
- **API Docs:** http://localhost:8000/docs
"""

    logger.info("Generated pipeline report")
    return report


@flow(
    name="daily-pipeline",
    description="Full daily data pipeline: ingest events, market data, run analysis",
    version="1.0.0",
)
def daily_pipeline(
    skip_events: bool = False,
    skip_market: bool = False,
    skip_analysis: bool = False,
    days_back_events: int = 30,
    days_back_market: int = 30,
    days_back_analysis: int = 30,
) -> dict:
    """
    Main daily pipeline orchestrating all data flows.

    This flow runs the complete data pipeline:
    1. Check prerequisites (database connection)
    2. Ingest GDELT events
    3. Ingest market data
    4. Run statistical analysis
    5. Generate summary report

    Parameters
    ----------
    skip_events : bool
        Skip event ingestion (default: False)
    skip_market : bool
        Skip market data ingestion (default: False)
    skip_analysis : bool
        Skip analysis (default: False)
    days_back_events : int
        Days of events to ingest (default: 30)
    days_back_market : int
        Days of market data to ingest (default: 30)
    days_back_analysis : int
        Days of data to analyze (default: 30)

    Returns
    -------
    dict
        Summary of all pipeline stages

    Example
    -------
    >>> # Run full pipeline
    >>> daily_pipeline()

    >>> # Run only analysis (skip ingestion)
    >>> daily_pipeline(skip_events=True, skip_market=True)
    """
    logger = get_run_logger()
    logger.info("Starting daily pipeline")

    # Check prerequisites
    prereq_check = check_prerequisites()
    if prereq_check.get("status") == "error":
        logger.error("Prerequisites check failed - aborting pipeline")
        return {"status": "failed", "stage": "prerequisites", "error": prereq_check.get("message")}

    # Stage 1: Ingest GDELT events
    if skip_events:
        logger.info("Skipping event ingestion")
        event_results = {"status": "skipped"}
    else:
        logger.info("Stage 1: Ingesting GDELT events")
        event_results = ingest_gdelt_events(
            days_back=days_back_events,
            skip_existing=True,
        )

    # Stage 2: Ingest market data
    if skip_market:
        logger.info("Skipping market data ingestion")
        market_results = {"status": "skipped"}
    else:
        logger.info("Stage 2: Ingesting market data")
        market_results = ingest_market_data(
            days_back=days_back_market,
            skip_fresh=True,
        )

    # Stage 3: Run analysis
    if skip_analysis:
        logger.info("Skipping analysis")
        analysis_results = {"status": "skipped"}
    else:
        logger.info("Stage 3: Running analysis")
        analysis_results = run_analysis(
            days_back=days_back_analysis,
            store_results=True,
        )

    # Generate report
    report = generate_pipeline_report(event_results, market_results, analysis_results)

    # Create artifact for Prefect UI
    create_markdown_artifact(
        key="daily-pipeline-report",
        markdown=report,
        description=f"Daily pipeline report for {date.today()}",
    )

    summary = {
        "status": "success",
        "date": str(date.today()),
        "stages": {
            "events": event_results,
            "market": market_results,
            "analysis": analysis_results,
        },
    }

    logger.info(f"Daily pipeline complete: {summary['status']}")
    return summary


def create_deployment():
    """
    Create Prefect deployment for production scheduling.

    This creates a deployment that:
    - Runs daily at 8 PM UTC
    - Ingests yesterday's events
    - Ingests last 5 days of market data
    - Analyzes last 7 days
    """
    from prefect.deployments import Deployment
    from prefect.server.schemas.schedules import CronSchedule

    deployment = Deployment.build_from_flow(
        flow=daily_pipeline,
        name="production",
        version="1.0.0",
        description="Production daily pipeline - runs at 8 PM UTC after markets close",
        schedule=CronSchedule(cron="0 20 * * *"),  # 8 PM UTC daily
        parameters={
            "days_back_events": 1,
            "days_back_market": 5,
            "days_back_analysis": 7,
        },
        tags=["production", "daily", "pipeline"],
    )

    deployment.apply()
    print("Deployment created: daily-pipeline/production")
    print("\nTo start the scheduler, run:")
    print("  prefect server start")
    print("\nThen in another terminal:")
    print("  prefect agent start -q 'default'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Daily Pipeline")
    parser.add_argument("--deploy", action="store_true", help="Create Prefect deployment")
    parser.add_argument("--skip-events", action="store_true", help="Skip event ingestion")
    parser.add_argument("--skip-market", action="store_true", help="Skip market ingestion")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip analysis")

    args = parser.parse_args()

    if args.deploy:
        create_deployment()
    else:
        result = daily_pipeline(
            skip_events=args.skip_events,
            skip_market=args.skip_market,
            skip_analysis=args.skip_analysis,
        )
        print(f"\nPipeline Result: {result['status']}")
