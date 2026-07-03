"""DuckDB connection + Postgres IO for the medallion transforms.

DuckDB is the compute engine; Postgres remains the storage layer. We ATTACH the
Postgres database so transforms read Bronze and write Silver with plain SQL —
no JVM, no JDBC driver, no cluster. This is the local-machine equivalent of
running DuckDB over a lakehouse; the SQL ports unchanged to MotherDuck/Snowflake.

Reads:  SELECT ... FROM pg.<bronze_table>
Writes: overwrite_table() replaces a Silver table's rows (DELETE + INSERT in a
        transaction) so the Alembic-defined schema and indexes are preserved.
"""

import logging
import os
from urllib.parse import urlparse

import duckdb

logger = logging.getLogger(__name__)

# Alias the attached Postgres database is mounted under inside DuckDB.
PG = "pg"

_DEFAULT_DB_URL = "postgresql://postgres:postgres@localhost:5432/geopolitical_tracker"


def _pg_conn_string() -> str:
    """Build a libpq connection string from DATABASE_URL for DuckDB's ATTACH.

    DuckDB's postgres extension wants the keyword form
    ("host=... port=... dbname=... user=... password=..."), not a URL.
    """
    url = os.getenv("DATABASE_URL", _DEFAULT_DB_URL)
    # Normalize SQLAlchemy-style driver suffixes to a plain postgres URL.
    for prefix in ("postgresql+psycopg2://", "postgresql+asyncpg://"):
        if url.startswith(prefix):
            url = "postgresql://" + url[len(prefix):]
    p = urlparse(url)
    return (
        f"host={p.hostname} port={p.port or 5432} "
        f"dbname={p.path.lstrip('/')} user={p.username} password={p.password}"
    )


def connect(attach_postgres: bool = True) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection, optionally with Postgres attached as `pg`.

    Args:
        attach_postgres: When True (default), ATTACH the Postgres DB so transforms
            can read `pg.<table>` and write Silver tables. Set False for offline
            work on Parquet/in-memory data (e.g. tests).
    """
    con = duckdb.connect()
    if attach_postgres:
        con.execute("INSTALL postgres; LOAD postgres;")
        con.execute(f"ATTACH '{_pg_conn_string()}' AS {PG} (TYPE postgres)")
    return con


def register_lookup(
    con: duckdb.DuckDBPyConnection,
    name: str,
    mapping: dict[str, str],
    key_col: str,
    val_col: str,
) -> None:
    """Materialize a Python dict as a DuckDB temp table for LEFT JOIN lookups."""
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE {name} "
        f"({key_col} VARCHAR, {val_col} VARCHAR)"
    )
    if mapping:
        con.executemany(
            f"INSERT INTO {name} VALUES (?, ?)", list(mapping.items())
        )


def overwrite_table(
    con: duckdb.DuckDBPyConnection,
    target_table: str,
    select_sql: str,
    columns: list[str],
) -> int:
    """Replace all rows in pg.<target_table> with the result of select_sql.

    DELETE + INSERT inside a transaction, so the table definition and indexes
    created by Alembic survive (unlike DROP/CREATE). Idempotent: re-running
    produces the same table state.

    Returns the row count written.
    """
    col_list = ", ".join(columns)
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(f"DELETE FROM {PG}.{target_table}")
        con.execute(
            f"INSERT INTO {PG}.{target_table} ({col_list}) {select_sql}"
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    (count,) = con.execute(
        f"SELECT count(*) FROM {PG}.{target_table}"
    ).fetchone()
    return count
