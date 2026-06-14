"""Silver Events Transform: Bronze events → silver_events.

Reads raw GDELT events from the Bronze 'events' table, applies:
- Deduplication on global_event_id
- CAMEO code → event_group classification (verbal_cooperation, material_conflict, etc.)
- Country code normalization (FIPS 10-4 → ISO 3166-1 alpha-3)
- Type casting and null handling
- is_significant flag (|goldstein| >= 5 AND mentions >= 1000)

Output: silver_events table — cleaned, typed, classified events ready for
downstream joins and Gold-layer aggregation.
"""

import logging
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, BooleanType

from src.config.constants import EVENT_GROUPS, FIPS_TO_ISO, CAMEO_CATEGORIES
from src.spark_transforms.spark_session import (
    get_spark_session,
    read_table,
    write_table,
)

logger = logging.getLogger(__name__)

# Significance thresholds — events with high impact + broad coverage
GOLDSTEIN_THRESHOLD = 5.0
MENTIONS_THRESHOLD = 1000


def _build_event_group_mapping() -> dict[str, str]:
    """Invert EVENT_GROUPS: CAMEO root code → group name."""
    mapping = {}
    for group_name, codes in EVENT_GROUPS.items():
        for code in codes:
            mapping[code] = group_name
    return mapping


# Pre-built mappings for broadcast variables
_EVENT_GROUP_MAP = _build_event_group_mapping()
_FIPS_TO_ISO_MAP = dict(FIPS_TO_ISO)
_CAMEO_LABEL_MAP = dict(CAMEO_CATEGORIES)


def transform_events(df: DataFrame) -> DataFrame:
    """Apply all Silver transformations to raw events.

    Args:
        df: Raw events DataFrame from Bronze table.

    Returns:
        Cleaned, classified events DataFrame ready for Silver table.
    """
    spark = get_spark_session()

    # Broadcast lookup maps for efficient joins on executors
    event_group_map = spark.sparkContext.broadcast(_EVENT_GROUP_MAP)
    fips_to_iso_map = spark.sparkContext.broadcast(_FIPS_TO_ISO_MAP)
    cameo_label_map = spark.sparkContext.broadcast(_CAMEO_LABEL_MAP)

    # UDFs using broadcast variables
    @F.udf(StringType())
    def classify_event_group(root_code):
        if root_code is None:
            return "other"
        code = str(root_code).zfill(2)[:2]
        return event_group_map.value.get(code, "other")

    @F.udf(StringType())
    def normalize_country_code(fips_code):
        if fips_code is None:
            return None
        return fips_to_iso_map.value.get(fips_code, fips_code)

    @F.udf(StringType())
    def get_cameo_label(root_code):
        if root_code is None:
            return "other"
        code = str(root_code).zfill(2)[:2]
        return cameo_label_map.value.get(code, "other")

    silver_df = (
        df
        # 1. Deduplicate on global_event_id (keep first occurrence)
        .dropDuplicates(["global_event_id"])

        # 2. Drop rows missing required fields
        .filter(
            F.col("global_event_id").isNotNull()
            & F.col("event_date").isNotNull()
            & F.col("event_root_code").isNotNull()
        )

        # 3. Classify events and normalize codes
        .withColumn("event_group", classify_event_group(F.col("event_root_code")))
        .withColumn("cameo_label", get_cameo_label(F.col("event_root_code")))
        .withColumn(
            "country_code",
            normalize_country_code(F.col("action_geo_country_code"))
        )

        # 4. Compute is_significant flag
        .withColumn(
            "is_significant",
            (
                (F.abs(F.col("goldstein_scale")) >= GOLDSTEIN_THRESHOLD)
                & (F.col("num_mentions") >= MENTIONS_THRESHOLD)
            ).cast(BooleanType())
        )

        # 5. Select and rename to Silver schema
        .select(
            F.col("global_event_id").alias("event_id"),
            F.col("event_date"),
            F.col("country_code"),
            F.col("event_group"),
            F.col("event_root_code"),
            F.col("cameo_label"),
            F.col("goldstein_scale"),
            F.col("num_mentions"),
            F.col("num_sources"),
            F.col("avg_tone"),
            F.col("actor1_name"),
            F.col("actor1_country_code").alias("actor1_country"),
            F.col("actor2_name"),
            F.col("actor2_country_code").alias("actor2_country"),
            F.col("action_geo_name").alias("geo_name"),
            F.col("action_geo_lat").alias("geo_lat"),
            F.col("action_geo_long").alias("geo_long"),
            F.col("is_significant"),
        )
    )

    return silver_df


def run():
    """Execute the full Silver events transform: read Bronze → transform → write Silver."""
    logger.info("Starting silver_events transform")

    # Read from Bronze
    bronze_df = read_table("events")
    logger.info(f"Bronze events: {bronze_df.count()} rows")

    # Transform
    silver_df = transform_events(bronze_df)

    # Write to Silver (idempotent — full overwrite)
    row_count = write_table(silver_df, "silver_events", mode="overwrite")
    logger.info(f"silver_events transform complete: {row_count} rows written")

    return row_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
