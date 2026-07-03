"""
Master Prefect flow: daily pipeline.

Runs three stages in sequence:
1. Ingestion — fetch fresh data from 5 sources into Bronze tables
2. Analysis — compute sentiment, embeddings, correlations on fresh data
3. Transforms — DuckDB Silver transforms + dbt Gold models (medallion architecture)

Scheduled via deploy.py to run daily at 6:00 AM UTC.
"""

from prefect import flow, get_run_logger

from flows.ingestion_flow import daily_ingestion
from flows.analysis_flow import daily_analysis
from flows.transform_flow import daily_transforms


@flow(name="daily-pipeline", log_prints=True)
def daily_pipeline():
    """Full daily pipeline: ingest → analyze → transform (medallion)."""
    logger = get_run_logger()
    logger.info("=== Starting Daily Pipeline ===")

    ingestion_result = daily_ingestion()
    logger.info(f"Ingestion complete: {ingestion_result}")

    analysis_result = daily_analysis()
    logger.info(f"Analysis complete: {analysis_result}")

    transform_result = daily_transforms()
    logger.info(f"Transforms complete: {transform_result}")

    logger.info("=== Daily Pipeline Complete ===")
    return {
        "ingestion": ingestion_result,
        "analysis": analysis_result,
        "transforms": transform_result,
    }


if __name__ == "__main__":
    daily_pipeline()
