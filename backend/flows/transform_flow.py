"""
Prefect flow for the medallion transform pipeline.

Runs PySpark Silver transforms first (Bronze → Silver), then dbt Gold models
(Silver → Gold). This is the third stage in the daily pipeline:

    ingestion → analysis → transforms (this flow)

Silver transforms run in dependency order:
1. silver_events (from Bronze events)
2. silver_market (from Bronze market_data)
3. silver_headlines (from Bronze news_headlines)
4. silver_event_market (from Silver events + Silver market — must run last)

Gold transforms run via dbt:
5. dbt run --select gold (builds all Gold models in dependency order)
6. dbt test --select gold (runs data quality tests)
"""

import subprocess
from pathlib import Path

from prefect import flow, task, get_run_logger

from src.spark_transforms.spark_session import stop_spark


# Path to the dbt project
DBT_PROJECT_DIR = Path(__file__).parent.parent / "dbt_project"


@task(name="silver-events", retries=1, log_prints=True)
def run_silver_events() -> int:
    """Transform Bronze events → Silver events."""
    from src.spark_transforms.silver_events import run
    return run()


@task(name="silver-market", retries=1, log_prints=True)
def run_silver_market() -> int:
    """Transform Bronze market_data → Silver market."""
    from src.spark_transforms.silver_market import run
    return run()


@task(name="silver-headlines", retries=1, log_prints=True)
def run_silver_headlines() -> int:
    """Transform Bronze news_headlines → Silver headlines."""
    from src.spark_transforms.silver_headlines import run
    return run()


@task(name="silver-event-market", retries=1, log_prints=True)
def run_silver_event_market() -> int:
    """Transform Silver events + Silver market → Silver event_market.

    Must run after silver_events and silver_market.
    """
    from src.spark_transforms.silver_event_market import run
    return run()


@task(name="stop-spark", log_prints=True)
def shutdown_spark():
    """Stop the SparkSession after all Silver transforms complete."""
    stop_spark()


@task(name="dbt-run-gold", retries=1, log_prints=True)
def run_dbt_gold() -> str:
    """Run dbt Gold models: Silver → Gold."""
    logger = get_run_logger()
    logger.info("Running dbt Gold models...")

    result = subprocess.run(
        ["dbt", "run", "--select", "gold", "--profiles-dir", str(DBT_PROJECT_DIR)],
        cwd=str(DBT_PROJECT_DIR),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"dbt run failed:\n{result.stderr}")
        raise RuntimeError(f"dbt run failed: {result.stderr}")

    logger.info(f"dbt run output:\n{result.stdout}")
    return result.stdout


@task(name="dbt-test-gold", log_prints=True)
def run_dbt_tests() -> str:
    """Run dbt tests on Gold models — data quality checks."""
    logger = get_run_logger()
    logger.info("Running dbt tests on Gold models...")

    result = subprocess.run(
        ["dbt", "test", "--select", "gold", "--profiles-dir", str(DBT_PROJECT_DIR)],
        cwd=str(DBT_PROJECT_DIR),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.warning(f"dbt test failures:\n{result.stdout}")
        # Don't raise — log warnings but don't fail the pipeline
        # Tests alert us to data quality issues, they don't block delivery

    logger.info(f"dbt test output:\n{result.stdout}")
    return result.stdout


@flow(name="daily-transforms", log_prints=True)
def daily_transforms() -> dict:
    """Full transform pipeline: PySpark Silver → dbt Gold.

    Dependency order:
    - silver_events, silver_market, silver_headlines run in parallel
    - silver_event_market runs after events + market complete
    - Spark shuts down after all Silver transforms
    - dbt Gold runs after Silver is complete
    - dbt tests run after Gold models are built
    """
    logger = get_run_logger()
    logger.info("=== Starting Medallion Transforms ===")

    # Stage 1: Independent Silver transforms (can run in parallel)
    events_count = run_silver_events()
    market_count = run_silver_market()
    headlines_count = run_silver_headlines()

    # Stage 2: Dependent Silver transform (needs events + market)
    event_market_count = run_silver_event_market(
        wait_for=[events_count, market_count]
    )

    # Stage 3: Shutdown Spark
    shutdown_spark(wait_for=[event_market_count, headlines_count])

    # Stage 4: dbt Gold models
    dbt_output = run_dbt_gold()

    # Stage 5: dbt data quality tests
    test_output = run_dbt_tests(wait_for=[dbt_output])

    silver_results = {
        "silver_events": events_count,
        "silver_market": market_count,
        "silver_headlines": headlines_count,
        "silver_event_market": event_market_count,
    }

    logger.info(f"Silver transforms: {silver_results}")
    logger.info("=== Medallion Transforms Complete ===")

    return {
        "silver": silver_results,
        "gold": "dbt run complete",
        "tests": "dbt tests complete",
    }


if __name__ == "__main__":
    daily_transforms()
