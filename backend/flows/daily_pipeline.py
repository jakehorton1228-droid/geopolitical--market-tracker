"""
Master Prefect flow: daily pipeline.

Runs ingestion first (fresh data), then analysis (compute on fresh data).
Scheduled via deploy.py to run daily at 6:00 AM UTC.
"""

from prefect import flow, get_run_logger

from flows.ingestion_flow import daily_ingestion
from flows.analysis_flow import daily_analysis


@flow(name="daily-pipeline", log_prints=True)
def daily_pipeline():
    """Full daily pipeline: ingest data, then run analysis."""
    logger = get_run_logger()
    logger.info("=== Starting Daily Pipeline ===")

    ingestion_result = daily_ingestion()
    logger.info(f"Ingestion complete: {ingestion_result}")

    analysis_result = daily_analysis()
    logger.info(f"Analysis complete: {analysis_result}")

    logger.info("=== Daily Pipeline Complete ===")
    return {"ingestion": ingestion_result, "analysis": analysis_result}


if __name__ == "__main__":
    daily_pipeline()
