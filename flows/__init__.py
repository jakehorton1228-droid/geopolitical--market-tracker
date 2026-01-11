"""
Prefect Flows for Geopolitical Market Tracker.

This module contains orchestration flows for:
- Data ingestion (GDELT events, market data)
- Statistical analysis (event studies, anomaly detection)
- Daily pipeline combining all stages

QUICK START:
------------
    # Run the daily pipeline manually
    python flows/daily_pipeline.py

    # Or import and run
    from flows.daily_pipeline import daily_pipeline
    daily_pipeline()

SCHEDULING:
-----------
    # 1. Start Prefect server (provides UI at http://localhost:4200)
    prefect server start

    # 2. In another terminal, create deployments
    python flows/daily_pipeline.py --deploy

    # 3. Start a worker to execute scheduled flows
    prefect worker start -p 'default-agent-pool'

AVAILABLE FLOWS:
----------------
    ingest_gdelt_events: Fetch and store GDELT events
    ingest_market_data:  Fetch and store Yahoo Finance data
    run_analysis:        Run event studies and anomaly detection
    daily_pipeline:      Full pipeline combining all above

For more details, see each flow's docstring or the README.
"""

from flows.ingest_events import ingest_gdelt_events
from flows.ingest_market import ingest_market_data
from flows.run_analysis import run_analysis
from flows.daily_pipeline import daily_pipeline

__all__ = [
    "ingest_gdelt_events",
    "ingest_market_data",
    "run_analysis",
    "daily_pipeline",
]
