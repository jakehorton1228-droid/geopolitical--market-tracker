"""Prefect flow for the enrichment stage — Python ML on freshly-loaded data.

Second stage of the daily pipeline:  ingest → enrich (this) → transform → analytics.

Enrichment is the ML work that can't be expressed as SQL, so it isn't part of the
dbt transform layer:
  * sentiment scoring (FinBERT) on new headlines
  * embeddings (sentence-transformers) for new headlines + events (RAG search)

It runs AFTER ingestion (there's new content to process) and BEFORE the dbt
transform — silver_headlines depends on sentiment_score being populated. Only rows
with NULL sentiment/embedding are processed, so it's fast when little is new.

This consolidates the sentiment/embedding logic that used to be duplicated across
ingestion_flow (post_ingest_nlp) and analysis_flow.
"""

from prefect import flow, task, get_run_logger

from src.analysis.sentiment import score_unprocessed_headlines
from src.analysis.embeddings import embed_unprocessed_headlines, embed_unprocessed_events


@task(name="score-sentiment", retries=1, retry_delay_seconds=30, log_prints=True)
def score_sentiment() -> dict:
    """FinBERT-score any headlines that don't yet have a sentiment_score."""
    logger = get_run_logger()
    logger.info("Scoring sentiment on unprocessed headlines")
    try:
        from src.db.connection import get_session
        with get_session() as session:
            n_scored = score_unprocessed_headlines(session)
        logger.info(f"Scored {n_scored} headlines")
        return {"headlines_scored": n_scored}
    except Exception as e:
        logger.warning(f"Sentiment scoring failed: {e}")
        return {"headlines_scored": 0, "error": str(e)}


@task(name="generate-embeddings", retries=1, retry_delay_seconds=30, log_prints=True)
def generate_embeddings() -> dict:
    """Embed any headlines/events that don't yet have an embedding (for RAG)."""
    logger = get_run_logger()
    logger.info("Generating embeddings for unprocessed content")
    try:
        from src.db.connection import get_session
        with get_session() as session:
            n_headlines = embed_unprocessed_headlines(session)
        with get_session() as session:
            n_events = embed_unprocessed_events(session)
        logger.info(f"Embedded {n_headlines} headlines, {n_events} events")
        return {"headlines_embedded": n_headlines, "events_embedded": n_events}
    except Exception as e:
        logger.warning(f"Embedding generation failed: {e}")
        return {"headlines_embedded": 0, "events_embedded": 0, "error": str(e)}


@flow(name="daily-enrich", log_prints=True)
def daily_enrich() -> dict:
    """Enrichment stage: sentiment + embeddings on newly-ingested content.

    Runs between ingestion and the dbt transform. Sentiment must be scored here
    because silver_headlines filters on sentiment_score being present.
    """
    logger = get_run_logger()
    logger.info("=== Starting Enrichment ===")

    sentiment_result = score_sentiment()
    embedding_result = generate_embeddings()

    logger.info("=== Enrichment Complete ===")
    return {"sentiment": sentiment_result, "embeddings": embedding_result}


if __name__ == "__main__":
    daily_enrich()
