"""Medallion Silver transforms (DuckDB).

Bronze (raw Postgres tables) → Silver (cleaned, typed, enriched Postgres tables).

DuckDB is the compute engine; Postgres stays the storage layer. At this data
scale (~1.5M rows) it's fast and lightweight — no JVM, no cluster — while the
medallion design and SQL port directly to a warehouse
(DuckDB → MotherDuck / Snowflake / BigQuery) if the data ever outgrows a node.

The Gold layer is built separately by dbt (Silver → Gold).
"""
