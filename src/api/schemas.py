"""
Pydantic schemas for API request/response validation.

WHY PYDANTIC?
-------------
Pydantic provides:
1. Automatic request validation (bad data returns 422 error)
2. Automatic response serialization (SQLAlchemy models -> JSON)
3. Auto-generated API documentation
4. Type hints that your IDE understands

SCHEMA NAMING CONVENTION:
------------------------
- *Base: Shared fields between create/read
- *Create: Fields needed to create a new record
- *Response: Fields returned from the API
- *Query: Query parameters for filtering
"""

from datetime import date, datetime
from decimal import Decimal
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal


# =============================================================================
# EVENT SCHEMAS
# =============================================================================

class EventBase(BaseModel):
    """Base event fields shared across schemas."""
    global_event_id: str
    event_date: date
    event_root_code: str
    event_base_code: str | None = None
    event_code: str | None = None

    actor1_code: str | None = None
    actor1_name: str | None = None
    actor1_country_code: str | None = None
    actor1_type: str | None = None

    actor2_code: str | None = None
    actor2_name: str | None = None
    actor2_country_code: str | None = None
    actor2_type: str | None = None

    is_root_event: bool = False
    goldstein_scale: float | None = None
    num_mentions: int | None = None
    num_sources: int | None = None
    num_articles: int | None = None
    avg_tone: float | None = None

    action_geo_country_code: str | None = None
    action_geo_name: str | None = None
    action_geo_lat: float | None = None
    action_geo_long: float | None = None
    source_url: str | None = None


class EventResponse(EventBase):
    """Event response with database ID and timestamps."""
    id: int
    created_at: datetime | None = None

    # This tells Pydantic to read attributes from SQLAlchemy models
    model_config = ConfigDict(from_attributes=True)


class EventQuery(BaseModel):
    """Query parameters for filtering events."""
    start_date: date | None = Field(None, description="Filter events from this date")
    end_date: date | None = Field(None, description="Filter events until this date")
    country_code: str | None = Field(None, description="3-letter ISO country code", max_length=3)
    event_root_codes: list[str] | None = Field(None, description="CAMEO root codes (01-20)")
    event_group: str | None = Field(None, description="Event group: verbal_cooperation, material_cooperation, verbal_conflict, material_conflict, violent_conflict")
    min_goldstein: float | None = Field(None, description="Minimum absolute Goldstein score")
    min_mentions: int | None = Field(None, description="Minimum number of mentions")
    limit: int = Field(100, ge=1, le=1000, description="Maximum results to return")
    offset: int = Field(0, ge=0, description="Number of results to skip")


# =============================================================================
# MARKET DATA SCHEMAS
# =============================================================================

class MarketDataBase(BaseModel):
    """Base market data fields."""
    symbol: str
    date: date
    open: Decimal | None = None
    high: Decimal | None = None
    low: Decimal | None = None
    close: Decimal
    adj_close: Decimal | None = None
    volume: int | None = None
    daily_return: float | None = None
    log_return: float | None = None


class MarketDataResponse(MarketDataBase):
    """Market data response with database ID."""
    id: int
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class MarketDataQuery(BaseModel):
    """Query parameters for market data."""
    symbol: str | None = Field(None, description="Market symbol (e.g., CL=F, SPY)")
    symbols: list[str] | None = Field(None, description="Multiple symbols")
    start_date: date | None = Field(None, description="Start date")
    end_date: date | None = Field(None, description="End date")
    limit: int = Field(1000, ge=1, le=10000, description="Maximum results")
    offset: int = Field(0, ge=0, description="Number of results to skip")


# =============================================================================
# ANALYSIS SCHEMAS
# =============================================================================

class AnalysisResultResponse(BaseModel):
    """Analysis result response."""
    id: int
    event_id: int
    symbol: str
    analysis_type: str

    # Event study results
    car: float | None = None
    car_t_stat: float | None = None
    car_p_value: float | None = None

    # Window details
    estimation_window_start: date | None = None
    estimation_window_end: date | None = None
    event_window_start: date | None = None
    event_window_end: date | None = None

    # Returns
    expected_return: float | None = None
    actual_return: float | None = None
    abnormal_return: float | None = None

    # Anomaly info
    is_anomaly: bool = False
    anomaly_type: str | None = None
    anomaly_score: float | None = None
    is_significant: bool = False

    created_at: datetime | None = None
    model_version: str | None = None

    model_config = ConfigDict(from_attributes=True)


class AnalysisQuery(BaseModel):
    """Query parameters for analysis results."""
    event_id: int | None = Field(None, description="Filter by event ID")
    symbol: str | None = Field(None, description="Filter by symbol")
    analysis_type: str | None = Field(None, description="event_study or anomaly_detection")
    is_significant: bool | None = Field(None, description="Only significant results")
    is_anomaly: bool | None = Field(None, description="Only anomalies")
    anomaly_type: str | None = Field(None, description="unexplained_move, muted_response, etc.")
    min_car: float | None = Field(None, description="Minimum absolute CAR")
    limit: int = Field(100, ge=1, le=1000)
    offset: int = Field(0, ge=0)


class AnomalyResponse(BaseModel):
    """Simplified anomaly response for anomaly endpoint."""
    id: int
    event_id: int
    event_date: date
    event_type: str
    symbol: str
    anomaly_type: str
    anomaly_score: float
    expected_return: float | None
    actual_return: float | None
    goldstein_scale: float | None

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# PREDICTION SCHEMAS
# =============================================================================

class PredictionRequest(BaseModel):
    """Request body for market direction prediction."""
    symbol: str = Field(..., description="Market symbol to predict")
    goldstein_scale: float = Field(..., ge=-10, le=10, description="Event Goldstein score")
    num_mentions: int = Field(1, ge=1, description="Number of media mentions")
    avg_tone: float = Field(0, ge=-10, le=10, description="Average media tone")
    event_root_code: str | None = Field(None, description="CAMEO root code")
    actor1_country_code: str | None = Field(None, description="Primary actor country")
    is_violent_conflict: bool = Field(False, description="Is violent conflict event")


class PredictionResponse(BaseModel):
    """Response from prediction endpoint."""
    symbol: str
    prediction: Literal["UP", "DOWN"]
    probability_up: float = Field(..., ge=0, le=1)
    probability_down: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1, description="How confident (0.5 = uncertain)")
    model_version: str = "demo-v1"
    disclaimer: str = "This is a demonstration model. Not financial advice."


# =============================================================================
# COMMON SCHEMAS
# =============================================================================

class PaginatedResponse(BaseModel):
    """Wrapper for paginated responses."""
    items: list
    total: int
    limit: int
    offset: int
    has_more: bool


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    database: str = "connected"
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str | None = None
    status_code: int
