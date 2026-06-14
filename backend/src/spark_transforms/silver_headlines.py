"""Silver Headlines Transform: Bronze news_headlines → silver_headlines.

Reads raw RSS headlines from the Bronze 'news_headlines' table, applies:
- Deduplication on URL (natural key)
- Source name normalization
- published_at → published_date extraction
- Filter to only headlines with sentiment scores (already processed by FinBERT)

Output: silver_headlines table — clean headlines with sentiment ready for
Gold-layer aggregation.
"""

import logging
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.spark_transforms.spark_session import (
    read_table,
    write_table,
)

logger = logging.getLogger(__name__)

# Canonical source names
SOURCE_NORMALIZATION = {
    "reuters": "reuters",
    "ap": "ap",
    "bbc": "bbc",
    "aljazeera": "aljazeera",
    "al jazeera": "aljazeera",
    "al_jazeera": "aljazeera",
}


def transform_headlines(df: DataFrame) -> DataFrame:
    """Apply all Silver transformations to raw headlines.

    Args:
        df: Raw news_headlines DataFrame from Bronze table.

    Returns:
        Cleaned headlines DataFrame with normalized fields.
    """
    # Build a mapping expression for source normalization
    source_mapping = F.create_map(
        *[item for pair in SOURCE_NORMALIZATION.items() for item in (F.lit(pair[0]), F.lit(pair[1]))]
    )

    silver_df = (
        df
        # 1. Deduplicate on URL (same article = same URL)
        .dropDuplicates(["url"])

        # 2. Drop rows missing required fields
        .filter(
            F.col("headline").isNotNull()
            & F.col("url").isNotNull()
            & F.col("published_at").isNotNull()
        )

        # 3. Only include headlines that have been sentiment-scored
        .filter(F.col("sentiment_score").isNotNull())

        # 4. Normalize source names
        .withColumn(
            "source",
            F.coalesce(
                source_mapping[F.lower(F.col("source"))],
                F.lower(F.col("source"))
            )
        )

        # 5. Extract date from published_at timestamp
        .withColumn("published_date", F.to_date(F.col("published_at")))

        # 6. Select Silver schema
        .select(
            F.col("id").alias("headline_id"),
            "source",
            "headline",
            "url",
            "published_date",
            "sentiment_score",
            "sentiment_label",
        )
    )

    return silver_df


def run():
    """Execute the full Silver headlines transform: read Bronze → transform → write Silver."""
    logger.info("Starting silver_headlines transform")

    # Read from Bronze
    bronze_df = read_table("news_headlines")
    logger.info(f"Bronze news_headlines: {bronze_df.count()} rows")

    # Transform
    silver_df = transform_headlines(bronze_df)

    # Write to Silver (idempotent — full overwrite)
    row_count = write_table(silver_df, "silver_headlines", mode="overwrite")
    logger.info(f"silver_headlines transform complete: {row_count} rows written")

    return row_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
