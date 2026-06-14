"""PySpark Silver layer transforms for the medallion architecture.

Bronze (raw tables) → Silver (cleaned, typed, joined) → Gold (dbt marts)

Each transform reads from Bronze tables via JDBC, applies cleaning/joining/
enrichment logic using PySpark DataFrames, and writes to Silver tables.
Runs in PySpark local mode — same API as Databricks, portable with zero changes.
"""
