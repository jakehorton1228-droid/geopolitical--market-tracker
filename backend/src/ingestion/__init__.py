"""
Data Ingestion Module.

This module handles fetching data from external sources:
- GDELT: Geopolitical events (conflicts, diplomacy, protests)
- Yahoo Finance: Market prices (stocks, commodities, currencies)
- RSS Feeds: News headlines (Reuters, AP, BBC, Al Jazeera)

USAGE:
------
    from src.ingestion import GDELTIngestion, MarketDataIngestion, RSSIngestion

    # Ingest geopolitical events
    gdelt = GDELTIngestion()
    gdelt.ingest_date(date(2024, 1, 15))

    # Ingest market data
    market = MarketDataIngestion()
    market.ingest_all_symbols(start_date, end_date)

    # Ingest news headlines
    rss = RSSIngestion()
    rss.ingest_all_feeds()
"""

from src.ingestion.gdelt import GDELTIngestion, fetch_sample_events
from src.ingestion.market_data import MarketDataIngestion, fetch_sample_prices
from src.ingestion.rss_feeds import RSSIngestion, fetch_sample_headlines

__all__ = [
    "GDELTIngestion",
    "MarketDataIngestion",
    "RSSIngestion",
    "fetch_sample_events",
    "fetch_sample_prices",
    "fetch_sample_headlines",
]
