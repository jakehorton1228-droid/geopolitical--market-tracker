"""Silver Event-Market Transform: silver_events + silver_market → silver_event_market.

The key cross-domain linking table. Joins geopolitical events to market reactions
using COUNTRY_ASSET_MAP — "when something happens in Russia, check Ruble, Gas, Oil."

Each row = one (event_date, country, symbol):
- What happened (aggregated event metrics: goldstein, mentions, dominant group)
- How markets reacted (close, returns, volatility on that date)

Depends on silver_events and silver_market (runs after both).
"""

import logging

from src.config.constants import COUNTRY_ASSET_MAP
from src.transforms.db import connect, overwrite_table

logger = logging.getLogger(__name__)

# Output columns, in silver_event_market table order (see Alembic migration).
SILVER_COLUMNS = [
    "event_date", "country_code", "symbol", "event_count",
    "avg_goldstein", "min_goldstein", "max_goldstein", "total_mentions",
    "avg_tone", "violent_count", "conflict_count", "cooperation_count",
    "dominant_event_group", "close", "daily_return", "log_return",
    "return_5d", "volatility_20d",
]


def register_lookups(con) -> None:
    """Materialize COUNTRY_ASSET_MAP ({country: [symbols]}) as a (country, symbol) table."""
    con.execute(
        "CREATE OR REPLACE TEMP TABLE country_asset "
        "(country_code VARCHAR, symbol VARCHAR)"
    )
    rows = [
        (country, symbol)
        for country, symbols in COUNTRY_ASSET_MAP.items()
        for symbol in symbols
    ]
    if rows:
        con.executemany("INSERT INTO country_asset VALUES (?, ?)", rows)


def build_select(
    events_source: str = "pg.silver_events",
    market_source: str = "pg.silver_market",
) -> str:
    """Return the SELECT that produces the Silver event-market rows.

    Aggregates events to one row per (date, country) to avoid join fan-out, maps
    each country to its tracked symbols, then joins to that symbol's market data
    on the same date.
    """
    return f"""
        WITH events_agg AS (
            SELECT
                event_date,
                country_code,
                count(*)                       AS event_count,
                avg(goldstein_scale)           AS avg_goldstein,
                min(goldstein_scale)           AS min_goldstein,
                max(goldstein_scale)           AS max_goldstein,
                sum(num_mentions)              AS total_mentions,
                avg(avg_tone)                  AS avg_tone,
                sum(CASE WHEN event_group = 'violent_conflict'
                         THEN 1 ELSE 0 END)    AS violent_count,
                sum(CASE WHEN event_group IN ('material_conflict', 'violent_conflict')
                         THEN 1 ELSE 0 END)    AS conflict_count,
                sum(CASE WHEN event_group IN ('verbal_cooperation', 'material_cooperation')
                         THEN 1 ELSE 0 END)    AS cooperation_count
            FROM {events_source}
            WHERE country_code IS NOT NULL
            GROUP BY event_date, country_code
        ),
        -- Dominant event group = most frequent per (date, country). Unlike a plain
        -- mode(), ties are broken DETERMINISTICALLY by severity (most impactful
        -- group wins) so the column is stable across runs and engines.
        grp_counts AS (
            SELECT event_date, country_code, event_group, count(*) AS gc
            FROM {events_source}
            WHERE country_code IS NOT NULL
            GROUP BY event_date, country_code, event_group
        ),
        dominant AS (
            SELECT event_date, country_code, event_group AS dominant_event_group
            FROM (
                SELECT event_date, country_code, event_group,
                       row_number() OVER (
                           PARTITION BY event_date, country_code
                           ORDER BY gc DESC,
                                    CASE event_group
                                        WHEN 'violent_conflict'     THEN 0
                                        WHEN 'material_conflict'    THEN 1
                                        WHEN 'verbal_conflict'      THEN 2
                                        WHEN 'material_cooperation' THEN 3
                                        WHEN 'verbal_cooperation'   THEN 4
                                        ELSE 5
                                    END
                       ) AS rn
                FROM grp_counts
            )
            WHERE rn = 1
        )
        SELECT
            ea.event_date            AS event_date,
            ea.country_code          AS country_code,
            ca.symbol                AS symbol,
            ea.event_count           AS event_count,
            ea.avg_goldstein         AS avg_goldstein,
            ea.min_goldstein         AS min_goldstein,
            ea.max_goldstein         AS max_goldstein,
            ea.total_mentions        AS total_mentions,
            ea.avg_tone              AS avg_tone,
            ea.violent_count         AS violent_count,
            ea.conflict_count        AS conflict_count,
            ea.cooperation_count     AS cooperation_count,
            d.dominant_event_group   AS dominant_event_group,
            m.close                  AS close,
            m.daily_return           AS daily_return,
            m.log_return             AS log_return,
            m.return_5d              AS return_5d,
            m.volatility_20d         AS volatility_20d
        FROM events_agg ea
        JOIN dominant d         ON ea.event_date = d.event_date
                               AND ea.country_code = d.country_code
        JOIN country_asset ca   ON ea.country_code = ca.country_code
        JOIN {market_source} m  ON m.symbol = ca.symbol
                               AND m.date   = ea.event_date
    """


def run() -> int:
    """Execute the Silver event-market transform: read Silver tables → write Silver."""
    logger.info("Starting silver_event_market transform")
    con = connect()
    try:
        register_lookups(con)
        row_count = overwrite_table(
            con,
            "silver_event_market",
            build_select("pg.silver_events", "pg.silver_market"),
            SILVER_COLUMNS,
        )
    finally:
        con.close()
    logger.info(f"silver_event_market transform complete: {row_count} rows written")
    return row_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
