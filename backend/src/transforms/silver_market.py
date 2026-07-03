"""Silver Market Transform: Bronze market_data → silver_market (DuckDB).

Reads raw OHLCV data from the Bronze 'market_data' table, applies:
- Deterministic deduplication on (symbol, date)
- Rolling 5-day / 20-day returns (window LAG)
- Rolling 20-day volatility (sample stddev of daily_return)
- Volume z-score (volume vs 20-day rolling mean/std)
- is_trading_day flag

Window functions here (LAG, rolling AVG/STDDEV with ROWS BETWEEN) are standard
SQL — the same partition/order/frame semantics any analytical warehouse exposes.
"""

import logging

from src.transforms.db import connect, overwrite_table

logger = logging.getLogger(__name__)

# Output columns, in silver_market table order (see Alembic migration).
SILVER_COLUMNS = [
    "symbol", "date", "open", "high", "low", "close", "volume",
    "daily_return", "log_return", "return_5d", "return_20d",
    "volatility_20d", "volume_zscore", "is_trading_day",
]


def build_select(source: str = "pg.market_data") -> str:
    """Return the SELECT that produces the Silver market rows.

    Args:
        source: Bronze relation — "pg.market_data" in production, or a temp view
            over a Parquet snapshot in tests.
    """
    return f"""
        WITH deduped AS (
            SELECT *,
                   row_number() OVER (
                       PARTITION BY symbol, date
                       ORDER BY volume DESC NULLS LAST
                   ) AS _rn
            FROM {source}
            WHERE symbol IS NOT NULL
              AND date   IS NOT NULL
              AND close  IS NOT NULL
        ),
        base AS (
            SELECT
                symbol, date,
                CAST(open   AS DOUBLE) AS open,
                CAST(high   AS DOUBLE) AS high,
                CAST(low    AS DOUBLE) AS low,
                CAST(close  AS DOUBLE) AS close,
                CAST(volume AS BIGINT) AS volume,
                daily_return, log_return
            FROM deduped
            WHERE _rn = 1
        ),
        windowed AS (
            SELECT *,
                   lag(close, 5)  OVER w  AS close_5d_ago,
                   lag(close, 20) OVER w  AS close_20d_ago,
                   stddev_samp(daily_return) OVER w20 AS volatility_20d,
                   avg(volume)               OVER w20 AS vol_mean_20d,
                   stddev_samp(volume)       OVER w20 AS vol_std_20d
            FROM base
            WINDOW
                w   AS (PARTITION BY symbol ORDER BY date),
                w20 AS (PARTITION BY symbol ORDER BY date
                        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
        )
        SELECT
            symbol, date, open, high, low, close, volume,
            daily_return, log_return,
            CASE WHEN close_5d_ago IS NOT NULL AND close_5d_ago <> 0
                 THEN (close - close_5d_ago) / close_5d_ago END  AS return_5d,
            CASE WHEN close_20d_ago IS NOT NULL AND close_20d_ago <> 0
                 THEN (close - close_20d_ago) / close_20d_ago END AS return_20d,
            volatility_20d,
            CASE WHEN vol_std_20d IS NOT NULL AND vol_std_20d > 0
                 THEN (CAST(volume AS DOUBLE) - vol_mean_20d) / vol_std_20d
                 END                                              AS volume_zscore,
            TRUE AS is_trading_day
        FROM windowed
    """


def run() -> int:
    """Execute the full Silver market transform: read Bronze → write Silver."""
    logger.info("Starting silver_market transform")
    con = connect()
    try:
        row_count = overwrite_table(
            con, "silver_market", build_select("pg.market_data"), SILVER_COLUMNS
        )
    finally:
        con.close()
    logger.info(f"silver_market transform complete: {row_count} rows written")
    return row_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
