"""Shared SparkSession factory and PostgreSQL JDBC helpers.

Provides a local-mode SparkSession and helper functions for reading/writing
PostgreSQL tables via JDBC. This is the same PySpark API used in Databricks —
code written here ports to a Databricks cluster with zero changes (just swap
the SparkSession builder and JDBC URL).
"""

import logging
import os
from pathlib import Path
from functools import lru_cache

from pyspark.sql import SparkSession, DataFrame

logger = logging.getLogger(__name__)

# JDBC driver — downloaded once, lives in backend/jars/
JARS_DIR = Path(__file__).parent.parent.parent / "jars"
JDBC_DRIVER = "org.postgresql.Driver"


def _find_jdbc_jar() -> str:
    """Locate the PostgreSQL JDBC driver jar."""
    if JARS_DIR.exists():
        jars = list(JARS_DIR.glob("postgresql-*.jar"))
        if jars:
            return str(jars[0])
    raise FileNotFoundError(
        f"PostgreSQL JDBC jar not found in {JARS_DIR}. "
        "Download from https://jdbc.postgresql.org/download/ "
        "and place in backend/jars/"
    )


@lru_cache(maxsize=1)
def get_spark_session() -> SparkSession:
    """Create or retrieve the shared SparkSession (local mode).

    Uses local[*] to leverage all available CPU cores — same as running
    on a single Databricks node. The JDBC jar is loaded for PostgreSQL access.
    """
    jdbc_jar = _find_jdbc_jar()

    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("gmip-silver-transforms")
        .config("spark.jars", jdbc_jar)
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")  # small data, fewer partitions
        .config("spark.ui.enabled", "false")  # no web UI in batch mode
        .getOrCreate()
    )

    # Reduce Spark's verbose logging
    spark.sparkContext.setLogLevel("WARN")
    logger.info("SparkSession created (local mode, driver memory=2g)")
    return spark


def _get_jdbc_url() -> str:
    """Build JDBC URL from DATABASE_URL environment variable."""
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://gmt:gmt_password@localhost:5432/geopolitical_market_tracker"
    )
    # SQLAlchemy uses postgresql://, JDBC uses jdbc:postgresql://
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "", 1)
    elif db_url.startswith("postgresql+psycopg2://"):
        db_url = db_url.replace("postgresql+psycopg2://", "", 1)

    # Parse user:password@host:port/dbname
    auth, rest = db_url.split("@", 1)
    user, password = auth.split(":", 1)
    host_port, dbname = rest.split("/", 1)

    jdbc_url = f"jdbc:postgresql://{host_port}/{dbname}"
    return jdbc_url, user, password


def read_table(table_name: str, spark: SparkSession = None) -> DataFrame:
    """Read a PostgreSQL table into a Spark DataFrame via JDBC.

    Args:
        table_name: The table to read (e.g., "events", "market_data").
        spark: Optional SparkSession. Uses shared session if not provided.

    Returns:
        Spark DataFrame with all rows from the table.
    """
    if spark is None:
        spark = get_spark_session()

    jdbc_url, user, password = _get_jdbc_url()

    df = (
        spark.read
        .format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", table_name)
        .option("user", user)
        .option("password", password)
        .option("driver", JDBC_DRIVER)
        .load()
    )
    logger.info(f"Read {df.count()} rows from {table_name}")
    return df


def read_query(query: str, spark: SparkSession = None) -> DataFrame:
    """Read a SQL query result into a Spark DataFrame via JDBC.

    Args:
        query: SQL query to execute (wrapped as a subquery for JDBC).
        spark: Optional SparkSession.

    Returns:
        Spark DataFrame with query results.
    """
    if spark is None:
        spark = get_spark_session()

    jdbc_url, user, password = _get_jdbc_url()

    df = (
        spark.read
        .format("jdbc")
        .option("url", jdbc_url)
        .option("query", query)
        .option("user", user)
        .option("password", password)
        .option("driver", JDBC_DRIVER)
        .load()
    )
    return df


def write_table(
    df: DataFrame,
    table_name: str,
    mode: str = "overwrite",
) -> int:
    """Write a Spark DataFrame to a PostgreSQL table via JDBC.

    Args:
        df: Spark DataFrame to write.
        table_name: Target table name.
        mode: Write mode — "overwrite" (truncate + insert) or "append".

    Returns:
        Row count written.
    """
    jdbc_url, user, password = _get_jdbc_url()
    row_count = df.count()

    (
        df.write
        .format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", table_name)
        .option("user", user)
        .option("password", password)
        .option("driver", JDBC_DRIVER)
        .mode(mode)
        .save()
    )
    logger.info(f"Wrote {row_count} rows to {table_name} (mode={mode})")
    return row_count


def stop_spark():
    """Stop the SparkSession. Call at end of pipeline run."""
    spark = get_spark_session()
    spark.stop()
    get_spark_session.cache_clear()
    logger.info("SparkSession stopped")
