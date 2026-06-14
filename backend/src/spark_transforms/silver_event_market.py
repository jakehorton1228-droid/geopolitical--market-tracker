"""Silver Event-Market Transform: silver_events + silver_market → silver_event_market.

The key cross-domain linking table. Joins geopolitical events to market reactions
using COUNTRY_ASSET_MAP — "when something happens in Russia, check Ruble, Gas, Oil."

Each row = one (event_date, country, symbol) combination, linking:
- What happened (event metrics: goldstein, mentions, event_group)
- How markets reacted (close, daily_return, log_return, return_5d)

This is the table that powers the Market sub-agent's correlation and pattern
analysis. It answers: "What was the market doing on the day of this event?"

Depends on: silver_events, silver_market (must run after both).
"""

import logging
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.config.constants import COUNTRY_ASSET_MAP
from src.spark_transforms.spark_session import (
    get_spark_session,
    read_table,
    write_table,
)

logger = logging.getLogger(__name__)


def _build_country_asset_rows() -> list[dict]:
    """Flatten COUNTRY_ASSET_MAP into rows for a Spark DataFrame.

    COUNTRY_ASSET_MAP: {"RUS": ["USDRUB=X", "NG=F", "CL=F"], ...}
    → [{"country_code": "RUS", "symbol": "USDRUB=X"}, ...]
    """
    rows = []
    for country, symbols in COUNTRY_ASSET_MAP.items():
        for symbol in symbols:
            rows.append({"country_code": country, "symbol": symbol})
    return rows


def transform_event_market(
    events_df: DataFrame,
    market_df: DataFrame,
) -> DataFrame:
    """Join events to market data via country-asset mapping.

    The join logic:
    1. Use COUNTRY_ASSET_MAP to know which symbols to check per country
    2. Join events → mapping → market on (country, symbol, date)
    3. Each output row links one event day + country to one symbol's market data

    Args:
        events_df: Silver events DataFrame.
        market_df: Silver market DataFrame.

    Returns:
        Joined DataFrame with event metrics + market reaction per day.
    """
    spark = get_spark_session()

    # Create the country→symbol mapping as a Spark DataFrame
    mapping_rows = _build_country_asset_rows()
    mapping_df = spark.createDataFrame(mapping_rows)

    # Aggregate events to one row per (date, country) to avoid fan-out
    events_agg = (
        events_df
        .filter(F.col("country_code").isNotNull())
        .groupBy("event_date", "country_code")
        .agg(
            F.count("*").alias("event_count"),
            F.avg("goldstein_scale").alias("avg_goldstein"),
            F.min("goldstein_scale").alias("min_goldstein"),
            F.max("goldstein_scale").alias("max_goldstein"),
            F.sum("num_mentions").alias("total_mentions"),
            F.avg("avg_tone").alias("avg_tone"),
            # Count by event group
            F.sum(
                F.when(F.col("event_group") == "violent_conflict", 1).otherwise(0)
            ).alias("violent_count"),
            F.sum(
                F.when(F.col("event_group").isin("material_conflict", "violent_conflict"), 1).otherwise(0)
            ).alias("conflict_count"),
            F.sum(
                F.when(F.col("event_group").isin("verbal_cooperation", "material_cooperation"), 1).otherwise(0)
            ).alias("cooperation_count"),
            # Dominant event group
            F.mode("event_group").alias("dominant_event_group"),
        )
    )

    # Join: events_agg → mapping → market
    # events_agg (date, country) → mapping (country → symbol) → market (symbol, date)
    silver_df = (
        events_agg
        .join(mapping_df, on="country_code", how="inner")
        .join(
            market_df.select(
                F.col("date").alias("market_date"),
                "symbol",
                "close",
                "daily_return",
                "log_return",
                "return_5d",
                "volatility_20d",
            ),
            on=[
                F.col("symbol") == market_df["symbol"],
                F.col("event_date") == F.col("market_date"),
            ],
            how="inner",
        )
        # Resolve ambiguous symbol column from the join
        .drop(market_df["symbol"])
        .drop("market_date")

        # Select final schema
        .select(
            "event_date",
            "country_code",
            "symbol",
            "event_count",
            "avg_goldstein",
            "min_goldstein",
            "max_goldstein",
            "total_mentions",
            "avg_tone",
            "violent_count",
            "conflict_count",
            "cooperation_count",
            "dominant_event_group",
            "close",
            "daily_return",
            "log_return",
            "return_5d",
            "volatility_20d",
        )
    )

    return silver_df


def run():
    """Execute the full Silver event-market transform.

    Depends on silver_events and silver_market being populated first.
    """
    logger.info("Starting silver_event_market transform")

    # Read from Silver tables (not Bronze — this is a Silver→Silver transform)
    events_df = read_table("silver_events")
    market_df = read_table("silver_market")
    logger.info(
        f"Input: {events_df.count()} silver events, "
        f"{market_df.count()} silver market rows"
    )

    # Transform
    silver_df = transform_event_market(events_df, market_df)

    # Write to Silver (idempotent — full overwrite)
    row_count = write_table(silver_df, "silver_event_market", mode="overwrite")
    logger.info(f"silver_event_market transform complete: {row_count} rows written")

    return row_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
