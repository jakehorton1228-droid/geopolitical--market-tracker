"""Prefect flow for the medallion transform layer — dbt on DuckDB.

The ENTIRE medallion (Bronze→Silver→Gold) is now a single dbt-duckdb project
(see backend/dbt_project/). This flow just drives dbt in three steps:

    seed  → load reference tables (CAMEO/FIPS/country-asset maps)
    run   → build every Silver + Gold model, in dependency order
    test  → data-quality checks (non-blocking)

Compute engine: DuckDB, attached to Postgres (the storage layer). This replaces
the old hand-written Python Silver transforms in src/transforms/ — dbt now owns
the whole transform layer, with one lineage graph, one set of tests, one engine.

It's the third stage of the daily pipeline:  ingest → enrich → transforms (this).
"""

import subprocess
from pathlib import Path

from prefect import flow, task, get_run_logger

# Path to the dbt project. profiles.yml there reads DATABASE_URL to attach Postgres.
DBT_PROJECT_DIR = Path(__file__).parent.parent / "dbt_project"


def _run_dbt(args: list[str]) -> subprocess.CompletedProcess:
    """Run a dbt subcommand inside the project dir, capturing output."""
    return subprocess.run(
        ["dbt", *args, "--profiles-dir", str(DBT_PROJECT_DIR)],
        cwd=str(DBT_PROJECT_DIR),
        capture_output=True,
        text=True,
    )


@task(name="dbt-seed", retries=1, log_prints=True)
def dbt_seed() -> str:
    """Load reference seeds into the DuckDB catalog.

    Seeds live in the ephemeral DuckDB file (not Postgres), so we (re)load them
    every run — cheap (a few hundred rows) and required before the models can
    join against them.
    """
    logger = get_run_logger()
    logger.info("dbt seed: loading reference tables")
    result = _run_dbt(["seed"])
    if result.returncode != 0:
        logger.error(f"dbt seed failed:\n{result.stdout}\n{result.stderr}")
        raise RuntimeError("dbt seed failed")
    logger.info(result.stdout)
    return result.stdout


@task(name="dbt-run", retries=1, log_prints=True)
def dbt_run() -> str:
    """Build the whole medallion — Silver + Gold — in dependency order."""
    logger = get_run_logger()
    logger.info("dbt run: building Silver + Gold models")
    result = _run_dbt(["run"])
    if result.returncode != 0:
        logger.error(f"dbt run failed:\n{result.stdout}\n{result.stderr}")
        raise RuntimeError("dbt run failed")
    logger.info(result.stdout)
    return result.stdout


@task(name="dbt-test", log_prints=True)
def dbt_test() -> str:
    """Run data-quality tests.

    Non-blocking by design: tests alert us to data-quality issues, they don't
    block delivery (same resilience principle the ingestion flow uses).
    """
    logger = get_run_logger()
    logger.info("dbt test: running data-quality checks")
    result = _run_dbt(["test"])
    if result.returncode != 0:
        logger.warning(f"dbt test failures (non-blocking):\n{result.stdout}")
    else:
        logger.info(result.stdout)
    return result.stdout


@flow(name="daily-transforms", log_prints=True)
def daily_transforms() -> dict:
    """Full medallion transform via dbt-duckdb: seed → run (Silver+Gold) → test."""
    logger = get_run_logger()
    logger.info("=== Starting Medallion Transforms (dbt-duckdb) ===")

    dbt_seed()
    dbt_run()
    dbt_test()

    logger.info("=== Medallion Transforms Complete ===")
    return {"status": "complete", "engine": "dbt-duckdb"}


if __name__ == "__main__":
    daily_transforms()
