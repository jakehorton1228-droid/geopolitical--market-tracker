"""SQLAlchemy database models for the geopolitical market tracker."""

from datetime import date, datetime
from decimal import Decimal
from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


class Event(Base):
    """
    GDELT geopolitical events.

    Each row represents a significant geopolitical event extracted from GDELT,
    such as military actions, diplomatic meetings, protests, etc.

    Key fields from GDELT:
    - global_event_id: GDELT's unique identifier
    - event_date: When the event occurred
    - event_root_code: CAMEO code category (01-20)
    - goldstein_scale: Impact score (-10 to +10)
    - num_mentions: How many articles mentioned this event
    - actor1/actor2: Countries or entities involved
    """
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    global_event_id = Column(String(50), unique=True, nullable=False, index=True)

    # Event timing
    event_date = Column(Date, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # CAMEO event classification
    event_root_code = Column(String(2), nullable=False, index=True)  # 01-20
    event_base_code = Column(String(4))  # More specific code
    event_code = Column(String(10))  # Full CAMEO code

    # Actors involved
    actor1_code = Column(String(10), index=True)  # e.g., "USA", "RUSGOV"
    actor1_name = Column(String(255))
    actor1_country_code = Column(String(3), index=True)  # ISO 3-letter
    actor1_type = Column(String(3))  # GOV, MIL, REB, etc.

    actor2_code = Column(String(10), index=True)
    actor2_name = Column(String(255))
    actor2_country_code = Column(String(3), index=True)
    actor2_type = Column(String(3))

    # Event metrics
    is_root_event = Column(Boolean, default=False)
    goldstein_scale = Column(Float)  # -10 (conflict) to +10 (cooperation)
    num_mentions = Column(Integer, default=1)
    num_sources = Column(Integer, default=1)
    num_articles = Column(Integer, default=1)
    avg_tone = Column(Float)  # Average sentiment of articles

    # Location
    action_geo_country_code = Column(String(3), index=True)
    action_geo_name = Column(String(255))
    action_geo_lat = Column(Float)
    action_geo_long = Column(Float)

    # Source
    source_url = Column(Text)

    # Relationships
    market_links = relationship("EventMarketLink", back_populates="event")
    analysis_results = relationship("AnalysisResult", back_populates="event")

    # Indexes for common queries
    __table_args__ = (
        Index("ix_events_date_country", "event_date", "action_geo_country_code"),
        Index("ix_events_date_code", "event_date", "event_root_code"),
        Index("ix_events_actors", "actor1_country_code", "actor2_country_code"),
    )

    def __repr__(self):
        return f"<Event {self.global_event_id}: {self.actor1_code} -> {self.actor2_code} ({self.event_root_code})>"


class MarketData(Base):
    """
    Daily market data for tracked financial instruments.

    Stores OHLCV data plus calculated returns for each symbol.
    Data sourced from Yahoo Finance.
    """
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)

    # Date
    date = Column(Date, nullable=False, index=True)

    # OHLCV data
    open = Column(Numeric(20, 6))
    high = Column(Numeric(20, 6))
    low = Column(Numeric(20, 6))
    close = Column(Numeric(20, 6), nullable=False)
    adj_close = Column(Numeric(20, 6))
    volume = Column(Integer)

    # Calculated fields (populated during ingestion or analysis)
    daily_return = Column(Float)  # (close - prev_close) / prev_close
    log_return = Column(Float)  # ln(close / prev_close)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    market_links = relationship("EventMarketLink", back_populates="market_data")

    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_market_data_symbol_date"),
        Index("ix_market_data_symbol_date", "symbol", "date"),
    )

    def __repr__(self):
        return f"<MarketData {self.symbol} {self.date}: {self.close}>"


class EventMarketLink(Base):
    """
    Links events to potentially affected market instruments.

    This is a many-to-many relationship table that connects geopolitical
    events to the financial instruments they might impact, based on the
    country-to-asset and event-to-asset mappings in constants.py.

    The 'relevance_score' indicates how strongly we expect this event
    to affect this particular asset (based on mapping rules).
    """
    __tablename__ = "event_market_links"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey("events.id", ondelete="CASCADE"), nullable=False)
    market_data_id = Column(Integer, ForeignKey("market_data.id", ondelete="CASCADE"), nullable=False)

    # How this link was determined
    link_type = Column(String(20))  # "country", "event_type", "manual"
    relevance_score = Column(Float, default=1.0)  # 0-1 score

    # Relationships
    event = relationship("Event", back_populates="market_links")
    market_data = relationship("MarketData", back_populates="market_links")

    __table_args__ = (
        UniqueConstraint("event_id", "market_data_id", name="uq_event_market_link"),
        Index("ix_event_market_link_event", "event_id"),
        Index("ix_event_market_link_market", "market_data_id"),
    )


class AnalysisResult(Base):
    """
    Stores results from event study and anomaly detection analysis.

    Each row represents an analysis of how a specific event impacted
    a specific market instrument, including:
    - Cumulative Abnormal Returns (CAR) from event study
    - Statistical significance (t-stat, p-value)
    - Whether this was flagged as an anomaly
    """
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey("events.id", ondelete="CASCADE"), nullable=False)
    symbol = Column(String(20), nullable=False, index=True)

    # Analysis type
    analysis_type = Column(String(30), nullable=False)  # "event_study", "anomaly_detection"

    # Event study results
    car = Column(Float)  # Cumulative Abnormal Return
    car_t_stat = Column(Float)  # T-statistic for CAR
    car_p_value = Column(Float)  # P-value

    # Window details
    estimation_window_start = Column(Date)
    estimation_window_end = Column(Date)
    event_window_start = Column(Date)
    event_window_end = Column(Date)

    # Expected vs actual
    expected_return = Column(Float)  # Based on market model
    actual_return = Column(Float)  # What actually happened
    abnormal_return = Column(Float)  # actual - expected

    # Anomaly detection
    is_anomaly = Column(Boolean, default=False, index=True)
    anomaly_type = Column(String(30))  # "unexplained_move", "muted_response", "delayed_reaction"
    anomaly_score = Column(Float)  # Z-score or similar

    # Significance
    is_significant = Column(Boolean, default=False, index=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String(20))  # Track which analysis model was used

    # Relationships
    event = relationship("Event", back_populates="analysis_results")

    __table_args__ = (
        UniqueConstraint("event_id", "symbol", "analysis_type", name="uq_analysis_result"),
        Index("ix_analysis_event_symbol", "event_id", "symbol"),
        Index("ix_analysis_anomalies", "is_anomaly", "is_significant"),
    )

    def __repr__(self):
        return f"<AnalysisResult {self.symbol} event={self.event_id}: CAR={self.car:.4f}>"
