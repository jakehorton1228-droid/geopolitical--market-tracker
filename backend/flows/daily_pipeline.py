"""Master Prefect flow: daily pipeline.

Four clean stages, each with one job and the right tool:

1. Ingest    — Python extract-load: fetch raw data from 5 sources into Bronze
2. Enrich    — Python ML: sentiment (FinBERT) + embeddings on new content
3. Transform — dbt on DuckDB: build the Silver + Gold medallion
4. Analytics — Python inference: correlations + historical patterns over Silver/Gold

Order matters: enrich runs before transform (silver_headlines needs sentiment),
and analytics runs after transform (it reads the freshly-built Silver layer).

Scheduled via deploy.py to run daily at 6:00 AM UTC.
"""

from prefect import flow, get_run_logger

from flows.ingestion_flow import daily_ingestion
from flows.enrich_flow import daily_enrich
from flows.transform_flow import daily_transforms
from flows.analysis_flow import daily_analysis


@flow(name="daily-pipeline", log_prints=True)
def daily_pipeline():
    """Full daily pipeline: ingest → enrich → transform → analytics."""
    logger = get_run_logger()
    logger.info("=== Starting Daily Pipeline ===")

    ingestion_result = daily_ingestion()
    logger.info(f"Ingestion complete: {ingestion_result}")

    enrich_result = daily_enrich()
    logger.info(f"Enrichment complete: {enrich_result}")

    transform_result = daily_transforms()
    logger.info(f"Transforms complete: {transform_result}")

    analysis_result = daily_analysis()
    logger.info(f"Analytics complete: {analysis_result}")

    logger.info("=== Daily Pipeline Complete ===")
    return {
        "ingestion": ingestion_result,
        "enrich": enrich_result,
        "transforms": transform_result,
        "analytics": analysis_result,
    }


if __name__ == "__main__":
    daily_pipeline()
