"""Silver Headlines Transform: Bronze news_headlines → silver_headlines (DuckDB).

Reads raw RSS headlines from the Bronze 'news_headlines' table, applies:
- Deduplication on URL (natural key)
- Source name normalization
- published_at → published_date extraction
- Filter to headlines that have a sentiment score (already scored by FinBERT)

Output: silver_headlines table — clean headlines with sentiment ready for
Gold-layer (dbt) aggregation.
"""

import logging

from src.transforms.db import connect, overwrite_table, register_lookup

logger = logging.getLogger(__name__)

# Canonical source names (raw lowercased → canonical).
SOURCE_NORMALIZATION = {
    "reuters": "reuters",
    "ap": "ap",
    "bbc": "bbc",
    "aljazeera": "aljazeera",
    "al jazeera": "aljazeera",
    "al_jazeera": "aljazeera",
}

# Output columns, in silver_headlines table order (see Alembic migration).
SILVER_COLUMNS = [
    "headline_id", "source", "headline", "url",
    "published_date", "sentiment_score", "sentiment_label",
]


def register_lookups(con) -> None:
    """Load the source-normalization lookup into the connection."""
    register_lookup(con, "source_norm", SOURCE_NORMALIZATION, "raw", "canonical")


def build_select(source: str = "pg.news_headlines") -> str:
    """Return the SELECT that produces the Silver headline rows.

    Args:
        source: Bronze relation — "pg.news_headlines" in production, or a temp
            view over a Parquet snapshot in tests.
    """
    return f"""
        WITH deduped AS (
            SELECT *,
                   row_number() OVER (
                       PARTITION BY url
                       ORDER BY published_at DESC NULLS LAST, id DESC
                   ) AS _rn
            FROM {source}
        )
        SELECT
            d.id                                       AS headline_id,
            COALESCE(sn.canonical, lower(d.source))    AS source,
            d.headline                                 AS headline,
            d.url                                      AS url,
            CAST(d.published_at AS DATE)               AS published_date,
            d.sentiment_score                          AS sentiment_score,
            d.sentiment_label                          AS sentiment_label
        FROM deduped d
        LEFT JOIN source_norm sn ON lower(d.source) = sn.raw
        WHERE d._rn = 1
          AND d.headline        IS NOT NULL
          AND d.url             IS NOT NULL
          AND d.published_at    IS NOT NULL
          AND d.sentiment_score IS NOT NULL
    """


def run() -> int:
    """Execute the full Silver headlines transform: read Bronze → write Silver."""
    logger.info("Starting silver_headlines transform")
    con = connect()
    try:
        register_lookups(con)
        row_count = overwrite_table(
            con, "silver_headlines", build_select("pg.news_headlines"), SILVER_COLUMNS
        )
    finally:
        con.close()
    logger.info(f"silver_headlines transform complete: {row_count} rows written")
    return row_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
