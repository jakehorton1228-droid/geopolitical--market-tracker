"""Database module for the geopolitical market tracker.

This module provides:
- SQLAlchemy models (Event, MarketData, AnalysisResult, EventMarketLink)
- Database connection utilities (engine, sessions)
- Query functions for common operations

Example usage:
    from src.db import get_session, Event, MarketData
    from src.db.queries import get_events_by_date_range

    with get_session() as session:
        events = get_events_by_date_range(session, start, end)
"""

from src.db.models import Base, Event, MarketData, AnalysisResult, EventMarketLink
from src.db.connection import engine, get_session, get_db, init_db, drop_db

__all__ = [
    # Models
    "Base",
    "Event",
    "MarketData",
    "AnalysisResult",
    "EventMarketLink",
    # Connection
    "engine",
    "get_session",
    "get_db",
    "init_db",
    "drop_db",
]
