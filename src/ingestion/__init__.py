"""
Data Ingestion Module.

This module handles fetching data from external sources:
- GDELT: Geopolitical events (conflicts, diplomacy, protests)
- Yahoo Finance: Market prices (stocks, commodities, currencies)

USAGE:
------
    from src.ingestion import GDELTIngestion, MarketDataIngestion

    # Ingest geopolitical events
    gdelt = GDELTIngestion()
    gdelt.ingest_date(date(2024, 1, 15))

    # Ingest market data
    market = MarketDataIngestion()
    market.ingest_all_symbols(start_date, end_date)
"""

from src.ingestion.gdelt import GDELTIngestion, fetch_sample_events
from src.ingestion.market_data import MarketDataIngestion, fetch_sample_prices

__all__ = [
    "GDELTIngestion",
    "MarketDataIngestion",
    "fetch_sample_events",
    "fetch_sample_prices",
]
