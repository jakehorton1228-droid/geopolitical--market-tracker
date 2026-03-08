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


class CorrelationCache(Base):
    """
    Pre-computed correlation results for fast dashboard loading.

    Populated by the daily Prefect pipeline. The API reads from this
    table instead of computing correlations on every request.
    One row per (symbol, event_metric, method) combination.
    """
    __tablename__ = "correlation_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    event_metric = Column(String(30), nullable=False)
    correlation = Column(Float, nullable=False)
    p_value = Column(Float, nullable=False)
    n_observations = Column(Integer, nullable=False)
    method = Column(String(10), nullable=False, default="pearson")
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    computed_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("symbol", "event_metric", "method", name="uq_correlation_cache"),
    )

class NewsHeadline(Base):
    """
    News headlines from RSS feeds (Reuters, AP, BBC, Al Jazeera).

    Ingested on a schedule to capture the news narrative around geopolitical
    events. Headlines are the raw input for sentiment analysis (Phase 2) —
    sentiment_score and sentiment_label remain NULL until then.

    The URL serves as the natural dedup key: same article = same URL.
    """
    __tablename__ = "news_headlines"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(20), nullable=False, index=True)  # "reuters", "ap", "bbc", "aljazeera"
    headline = Column(Text, nullable=False)
    url = Column(Text, nullable=False, unique=True)
    description = Column(Text, nullable=True)  # RSS summary, can be long or absent
    published_at = Column(DateTime, nullable=False, index=True)

    # Sentiment — populated in Phase 2 by analysis/sentiment.py
    sentiment_score = Column(Float, nullable=True)  # -1.0 (negative) to +1.0 (positive)
    sentiment_label = Column(String(10), nullable=True)  # "positive", "negative", "neutral"

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_news_headlines_source_published", "source", "published_at"),
    )

    def __repr__(self):
        return f"<NewsHeadline {self.source}: {self.headline[:50]}>"


class EconomicIndicator(Base):
    """
    Economic indicators from FRED (Federal Reserve Economic Data).

    Each row is a single observation: one value for one series on one date.
    For example: GDP = 27610.1 on 2024-01-01, or UNRATE = 3.7 on 2024-02-01.

    Series have different frequencies (daily, monthly, quarterly) but we store
    them all the same way — the date is the observation date FRED reports.

    The (series_id, date) pair is the natural dedup key.
    """
    __tablename__ = "economic_indicators"

    id = Column(Integer, primary_key=True, autoincrement=True)
    series_id = Column(String(20), nullable=False, index=True)  # "GDP", "CPIAUCSL", etc.
    series_name = Column(String(100), nullable=False)  # Human-readable name
    date = Column(Date, nullable=False, index=True)
    value = Column(Float, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("series_id", "date", name="uq_economic_indicator_series_date"),
        Index("ix_economic_indicators_series_date", "series_id", "date"),
    )

    def __repr__(self):
        return f"<EconomicIndicator {self.series_id} {self.date}: {self.value}>"


class PredictionMarket(Base):
    """
    Prediction market snapshots from Polymarket.

    Each row is a point-in-time snapshot of a market's probability. We store
    one snapshot per market per day. Tracking daily snapshots lets us see how
    crowd-estimated probabilities shift over time — a leading indicator of
    geopolitical risk that complements our backward-looking event data.

    The yes_price IS the implied probability: $0.35 = 35% chance of "Yes".

    The (market_id, snapshot_date) pair is the dedup key — one snapshot per
    market per day.
    """
    __tablename__ = "prediction_markets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(String(20), nullable=False, index=True)  # Polymarket's ID
    question = Column(Text, nullable=False)  # "Will Russia and Ukraine reach a ceasefire?"
    event_title = Column(String(255), nullable=True)  # Parent event grouping

    # The key data: probability and trading activity
    yes_price = Column(Float, nullable=False)  # 0.0-1.0, this IS the probability
    volume_24h = Column(Float, nullable=True)  # USD volume in last 24h
    total_volume = Column(Float, nullable=True)  # Lifetime USD volume
    liquidity = Column(Float, nullable=True)  # Current liquidity pool

    # Market metadata
    end_date = Column(String(30), nullable=True)  # When the market resolves

    # Snapshot timing
    snapshot_at = Column(DateTime, nullable=False, index=True)
    snapshot_date = Column(Date, nullable=False, index=True)  # Derived from snapshot_at for dedup

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("market_id", "snapshot_date", name="uq_prediction_market_snapshot"),
        Index("ix_prediction_markets_market_date", "market_id", "snapshot_date"),
    )

    def __repr__(self):
        return f"<PredictionMarket {self.market_id}: {self.question[:50]} @ {self.yes_price}>"