"""Silver Market Transform: Bronze market_data → silver_market.

Reads raw OHLCV data from the Bronze 'market_data' table, applies:
- Deduplication on (symbol, date)
- Rolling return calculations (5-day, 20-day) via Spark Window functions
- Rolling volatility (20-day standard deviation of daily returns)
- Volume z-score (volume relative to 20-day rolling mean/std)
- is_trading_day flag

Window functions here are identical to what you'd write in Databricks —
same pyspark.sql.Window API, same partitionBy/orderBy patterns.
"""

import logging
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.spark_transforms.spark_session import (
    read_table,
    write_table,
)

logger = logging.getLogger(__name__)


def transform_market(df: DataFrame) -> DataFrame:
    """Apply all Silver transformations to raw market data.

    Uses Spark Window functions for rolling calculations — the same API
    used in Databricks notebooks and Foundry transforms.

    Args:
        df: Raw market_data DataFrame from Bronze table.

    Returns:
        Enriched market DataFrame with rolling metrics.
    """
    # Window: per-symbol, ordered by date
    symbol_window = Window.partitionBy("symbol").orderBy("date")

    # Rolling windows for 5-day and 20-day calculations
    rolling_5d = symbol_window.rowsBetween(-4, 0)   # current row + 4 prior
    rolling_20d = symbol_window.rowsBetween(-19, 0)  # current row + 19 prior

    silver_df = (
        df
        # 1. Deduplicate on (symbol, date)
        .dropDuplicates(["symbol", "date"])

        # 2. Drop rows missing required fields
        .filter(
            F.col("symbol").isNotNull()
            & F.col("date").isNotNull()
            & F.col("close").isNotNull()
        )

        # 3. Cast close to double (it's Numeric in Bronze)
        .withColumn("close", F.col("close").cast("double"))
        .withColumn("open", F.col("open").cast("double"))
        .withColumn("high", F.col("high").cast("double"))
        .withColumn("low", F.col("low").cast("double"))

        # 4. Rolling 5-day return: (close - close_5d_ago) / close_5d_ago
        .withColumn(
            "close_5d_ago",
            F.lag("close", 5).over(symbol_window)
        )
        .withColumn(
            "return_5d",
            F.when(
                F.col("close_5d_ago").isNotNull() & (F.col("close_5d_ago") != 0),
                (F.col("close") - F.col("close_5d_ago")) / F.col("close_5d_ago")
            )
        )

        # 5. Rolling 20-day return
        .withColumn(
            "close_20d_ago",
            F.lag("close", 20).over(symbol_window)
        )
        .withColumn(
            "return_20d",
            F.when(
                F.col("close_20d_ago").isNotNull() & (F.col("close_20d_ago") != 0),
                (F.col("close") - F.col("close_20d_ago")) / F.col("close_20d_ago")
            )
        )

        # 6. Rolling 20-day volatility (std dev of daily returns)
        .withColumn(
            "volatility_20d",
            F.stddev("daily_return").over(rolling_20d)
        )

        # 7. Volume z-score: (volume - rolling_mean) / rolling_std
        .withColumn("vol_mean_20d", F.avg("volume").over(rolling_20d))
        .withColumn("vol_std_20d", F.stddev("volume").over(rolling_20d))
        .withColumn(
            "volume_zscore",
            F.when(
                F.col("vol_std_20d").isNotNull() & (F.col("vol_std_20d") > 0),
                (F.col("volume") - F.col("vol_mean_20d")) / F.col("vol_std_20d")
            )
        )

        # 8. is_trading_day flag (true for all rows — non-trading days aren't ingested)
        .withColumn("is_trading_day", F.lit(True))

        # 9. Select Silver schema (drop intermediate columns)
        .select(
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            F.col("volume").cast("long").alias("volume"),
            "daily_return",
            "log_return",
            "return_5d",
            "return_20d",
            "volatility_20d",
            "volume_zscore",
            "is_trading_day",
        )
    )

    return silver_df


def run():
    """Execute the full Silver market transform: read Bronze → transform → write Silver."""
    logger.info("Starting silver_market transform")

    # Read from Bronze
    bronze_df = read_table("market_data")
    logger.info(f"Bronze market_data: {bronze_df.count()} rows")

    # Transform
    silver_df = transform_market(bronze_df)

    # Write to Silver (idempotent — full overwrite)
    row_count = write_table(silver_df, "silver_market", mode="overwrite")
    logger.info(f"silver_market transform complete: {row_count} rows written")

    return row_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
