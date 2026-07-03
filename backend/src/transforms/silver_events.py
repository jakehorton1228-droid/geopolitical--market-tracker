"""Silver Events Transform: Bronze events → silver_events (DuckDB).

Reads raw GDELT events from the Bronze 'events' table, applies:
- Deterministic deduplication on global_event_id (keep the best-covered row)
- CAMEO code → event_group classification (verbal_cooperation, material_conflict, ...)
- Country code normalization (FIPS 10-4 → ISO 3166-1 alpha-3)
- is_significant flag (|goldstein| >= 5 AND mentions >= 1000)

Output: silver_events table — cleaned, typed, classified events ready for
downstream joins and Gold-layer (dbt) aggregation.
"""

import logging

from src.transforms.db import connect, overwrite_table, register_lookup
from src.transforms.mappings import (
    EVENT_GROUP_BY_CODE,
    CAMEO_LABEL_BY_CODE,
    FIPS_BY_CODE,
)

logger = logging.getLogger(__name__)

# Significance thresholds — high impact + broad coverage
GOLDSTEIN_THRESHOLD = 5.0
MENTIONS_THRESHOLD = 1000

# Output columns, in silver_events table order (see Alembic migration).
SILVER_COLUMNS = [
    "event_id", "event_date", "country_code", "event_group", "event_root_code",
    "cameo_label", "goldstein_scale", "num_mentions", "num_sources", "avg_tone",
    "actor1_name", "actor1_country", "actor2_name", "actor2_country",
    "geo_name", "geo_lat", "geo_long", "is_significant",
]


def register_lookups(con) -> None:
    """Load the CAMEO/FIPS lookup dicts into the connection as temp tables."""
    register_lookup(con, "event_group_map", EVENT_GROUP_BY_CODE, "code", "grp")
    register_lookup(con, "cameo_label_map", CAMEO_LABEL_BY_CODE, "code", "label")
    register_lookup(con, "fips_iso_map", FIPS_BY_CODE, "fips", "iso")


def build_select(source: str = "pg.events") -> str:
    """Return the SELECT that produces the Silver events rows.

    Args:
        source: The Bronze relation to read from — "pg.events" in production,
            or a temp view over a Parquet snapshot in tests.
    """
    return f"""
        WITH deduped AS (
            SELECT *,
                   row_number() OVER (
                       PARTITION BY global_event_id
                       ORDER BY num_mentions DESC NULLS LAST,
                                num_sources  DESC NULLS LAST
                   ) AS _rn
            FROM {source}
            WHERE global_event_id  IS NOT NULL
              AND event_date       IS NOT NULL
              AND event_root_code  IS NOT NULL
        )
        SELECT
            d.global_event_id                              AS event_id,
            d.event_date                                   AS event_date,
            COALESCE(f.iso, d.action_geo_country_code)     AS country_code,
            COALESCE(g.grp, 'other')                       AS event_group,
            d.event_root_code                              AS event_root_code,
            COALESCE(c.label, 'other')                     AS cameo_label,
            d.goldstein_scale                              AS goldstein_scale,
            d.num_mentions                                 AS num_mentions,
            d.num_sources                                  AS num_sources,
            d.avg_tone                                     AS avg_tone,
            d.actor1_name                                  AS actor1_name,
            d.actor1_country_code                          AS actor1_country,
            d.actor2_name                                  AS actor2_name,
            d.actor2_country_code                          AS actor2_country,
            d.action_geo_name                              AS geo_name,
            d.action_geo_lat                               AS geo_lat,
            d.action_geo_long                              AS geo_long,
            (ABS(d.goldstein_scale) >= {GOLDSTEIN_THRESHOLD}
                 AND d.num_mentions >= {MENTIONS_THRESHOLD}) AS is_significant
        FROM deduped d
        LEFT JOIN event_group_map g
               ON substr(lpad(d.event_root_code, 2, '0'), 1, 2) = g.code
        LEFT JOIN cameo_label_map c
               ON substr(lpad(d.event_root_code, 2, '0'), 1, 2) = c.code
        LEFT JOIN fips_iso_map f
               ON d.action_geo_country_code = f.fips
        WHERE d._rn = 1
    """


def run() -> int:
    """Execute the full Silver events transform: read Bronze → write Silver."""
    logger.info("Starting silver_events transform")
    con = connect()
    try:
        register_lookups(con)
        row_count = overwrite_table(
            con, "silver_events", build_select("pg.events"), SILVER_COLUMNS
        )
    finally:
        con.close()
    logger.info(f"silver_events transform complete: {row_count} rows written")
    return row_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
