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
# PREDICTION SCHEMAS (Logistic Regression - Level 2)
# =============================================================================

class LogisticPredictionRequest(BaseModel):
    """Request body for logistic regression prediction."""
    symbol: str = Field(..., description="Market symbol to predict")
    goldstein_mean: float = Field(0.0, description="Mean Goldstein score of today's events")
    goldstein_min: float = Field(0.0, description="Min Goldstein score")
    goldstein_max: float = Field(0.0, description="Max Goldstein score")
    mentions_total: int = Field(0, ge=0, description="Total media mentions")
    avg_tone: float = Field(0.0, description="Average media tone")
    conflict_count: int = Field(0, ge=0, description="Number of conflict events")
    cooperation_count: int = Field(0, ge=0, description="Number of cooperation events")
    training_days: int = Field(365, ge=90, le=730, description="Days of training data")


class LogisticPredictionResponse(BaseModel):
    """Response from logistic regression prediction."""
    symbol: str
    prediction: Literal["UP", "DOWN"]
    probability_up: float = Field(..., ge=0, le=1)
    probability_down: float = Field(..., ge=0, le=1)
    accuracy: float = Field(..., ge=0, le=1, description="Cross-validated accuracy")
    n_training_samples: int
    feature_contributions: list[dict]
    coefficients: dict[str, float]
    disclaimer: str = "Statistical model for educational purposes. Not financial advice."


# =============================================================================
# HISTORICAL PATTERN SCHEMAS (Level 1)
# =============================================================================

class HistoricalPatternResponse(BaseModel):
    """Response from historical frequency pattern analysis."""
    symbol: str
    event_filter: str
    total_occurrences: int
    up_count: int
    down_count: int
    up_percentage: float
    avg_return_up: float
    avg_return_down: float
    avg_return_all: float
    median_return: float
    t_statistic: float
    p_value: float
    is_significant: bool


# =============================================================================
# CORRELATION SCHEMAS
# =============================================================================

class CorrelationResponse(BaseModel):
    """Response from correlation analysis."""
    symbol: str
    event_metric: str
    correlation: float
    p_value: float
    n_observations: int
    method: str


# =============================================================================
# EVENT STUDY SCHEMAS
# =============================================================================

class EventStudyRequest(BaseModel):
    """Request parameters for running an event study."""
    symbol: str = Field(..., description="Market symbol (e.g., CL=F)")
    event_id: int = Field(..., description="Event ID to analyze")
    event_date: date = Field(..., description="Date of the event")


class EventStudyResponse(BaseModel):
    """Response from event study analysis."""
    event_id: int
    symbol: str
    event_date: date
    car: float = Field(..., description="Cumulative Abnormal Return")
    car_percent: float = Field(..., description="CAR as percentage")
    t_statistic: float
    p_value: float
    is_significant: bool
    ci_lower: float
    ci_upper: float
    expected_return: float
    actual_return: float
    std_dev: float
    wilcoxon_p: float | None = None
    estimation_days: int
    event_days: int
    summary: str


# =============================================================================
# REGRESSION SCHEMAS
# =============================================================================

class RegressionResponse(BaseModel):
    """Response from regression analysis."""
    symbol: str
    r_squared: float
    adj_r_squared: float
    f_statistic: float
    f_pvalue: float
    coefficients: dict[str, float]
    std_errors: dict[str, float]
    t_values: dict[str, float]
    p_values: dict[str, float]
    conf_int_lower: dict[str, float]
    conf_int_upper: dict[str, float]
    n_observations: int
    n_features: int
    summary: str


# =============================================================================
# ANOMALY DETECTION SCHEMAS
# =============================================================================

class AnomalyDetectionResponse(BaseModel):
    """Response from production anomaly detection."""
    date: date
    symbol: str
    anomaly_type: str
    actual_return: float
    expected_return: float
    z_score: float
    isolation_score: float
    anomaly_probability: float
    event_count: int = 0
    avg_goldstein: float = 0.0
    detected_by: list[str] = []


class AnomalyReportResponse(BaseModel):
    """Summary report from anomaly detection."""
    symbol: str
    start_date: date
    end_date: date
    total_days: int
    anomaly_count: int
    anomaly_rate: float
    unexplained_moves: int
    muted_responses: int
    statistical_outliers: int
    top_anomalies: list[AnomalyDetectionResponse]
    summary: str


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


# =============================================================================
# ECONOMIC INDICATOR SCHEMAS
# =============================================================================

class IndicatorResponse(BaseModel):
    """Single FRED indicator observation."""
    series_id: str
    series_name: str
    date: date
    value: float

    model_config = ConfigDict(from_attributes=True)


class IndicatorWithDeltaResponse(BaseModel):
    """Latest FRED indicator with change from previous observation."""
    series_id: str
    series_name: str
    date: date
    value: float
    previous_value: float | None = None
    previous_date: date | None = None
    change: float | None = None
    change_pct: float | None = None


# =============================================================================
# NEWS HEADLINE SCHEMAS
# =============================================================================

class HeadlineResponse(BaseModel):
    """News headline from RSS feeds."""
    id: int
    source: str
    headline: str
    url: str
    description: str | None = None
    published_at: datetime
    sentiment_score: float | None = None
    sentiment_label: str | None = None

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# PREDICTION MARKET SCHEMAS
# =============================================================================

class PredictionMarketResponse(BaseModel):
    """Prediction market snapshot from Polymarket."""
    id: int
    market_id: str
    question: str
    event_title: str | None = None
    yes_price: float
    volume_24h: float | None = None
    total_volume: float | None = None
    liquidity: float | None = None
    end_date: str | None = None
    snapshot_date: date

    model_config = ConfigDict(from_attributes=True)


class PredictionMoverResponse(BaseModel):
    """Prediction market with probability change over a time window."""
    market_id: str
    question: str
    event_title: str | None = None
    current_price: float
    previous_price: float
    price_change: float
    abs_change: float
    snapshot_date: date
    previous_date: date


# =============================================================================
# EVENT AGGREGATE SCHEMAS (typed replacements for former dict responses)
# =============================================================================

class EventCountResponse(BaseModel):
    """Total event count for a date range."""
    count: int
    start_date: date
    end_date: date


class CountryEventCountResponse(BaseModel):
    """Event count and average Goldstein score for a single country."""
    country_code: str
    count: int
    avg_goldstein: float | None = None


class EventTypeCountResponse(BaseModel):
    """Event count for a single CAMEO type."""
    code: str
    name: str
    group: str
    count: int


class EventMapResponse(BaseModel):
    """Per-country event aggregates for the world-map visualization."""
    country_code: str
    event_count: int
    avg_goldstein: float
    total_mentions: int
    conflict_count: int
    cooperation_count: int


# =============================================================================
# MARKET AGGREGATE SCHEMAS
# =============================================================================

class ReturnPointResponse(BaseModel):
    """A single (date, return) point for a symbol's return series."""
    date: date
    return_pct: float


class SymbolStatsResponse(BaseModel):
    """Summary statistics for a symbol over a date range."""
    symbol: str
    name: str
    category: str
    start_date: date
    end_date: date
    data_points: int
    mean_daily_return_pct: float | None = None
    min_return_pct: float | None = None
    max_return_pct: float | None = None
    min_price: float | None = None
    max_price: float | None = None


# =============================================================================
# ANALYSIS LIST SCHEMAS
# =============================================================================

class AnomalyListItem(BaseModel):
    """A detected anomaly joined with its underlying event."""
    id: int
    event_id: int
    event_date: date
    event_type: str
    event_group: str
    symbol: str
    anomaly_type: str | None = None
    anomaly_score: float | None = None
    expected_return: float | None = None
    actual_return: float | None = None
    goldstein_scale: float | None = None
    actor1: str | None = None
    actor2: str | None = None


class SignificantResultItem(BaseModel):
    """A statistically significant event-study result joined with its event."""
    event_id: int
    event_date: date
    event_type: str
    symbol: str
    car_pct: float
    t_stat: float | None = None
    p_value: float | None = None
    goldstein_scale: float | None = None
    num_mentions: int | None = None


class AnalysisSummaryResponse(BaseModel):
    """Aggregate counts across all analysis results."""
    total_results: int
    significant_results: int
    significance_rate: float
    total_anomalies: int
    anomaly_breakdown: dict[str, int]


# =============================================================================
# ROLLING CORRELATION SCHEMAS
# =============================================================================

class RollingCorrelationPoint(BaseModel):
    """One point in a rolling-correlation time series."""
    date: str
    correlation: float
    upper_ci: float
    lower_ci: float


class RollingCorrelationResponse(BaseModel):
    """Rolling-window correlation between an event metric and returns."""
    symbol: str
    event_metric: str
    window_days: int
    data: list[RollingCorrelationPoint]
