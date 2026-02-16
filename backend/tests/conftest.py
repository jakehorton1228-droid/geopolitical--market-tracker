"""
Shared test fixtures for the geopolitical market tracker.

Provides mock database sessions, sample data factories, and
common test utilities used across all test modules.
"""

import sys
from pathlib import Path
from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Sample data factories
# ---------------------------------------------------------------------------

def make_event(
    id: int = 1,
    event_date: date = None,
    event_root_code: str = "14",
    goldstein_scale: float = -3.0,
    num_mentions: int = 25,
    avg_tone: float = -2.5,
    actor1_code: str = "USA",
    actor2_code: str = "RUS",
    actor1_country_code: str = "USA",
    actor2_country_code: str = "RUS",
    action_geo_country_code: str = "UKR",
    actor1_name: str = "United States",
    actor2_name: str = "Russia",
    global_event_id: str = None,
):
    """Create a mock Event object."""
    if event_date is None:
        event_date = date.today() - timedelta(days=5)
    if global_event_id is None:
        global_event_id = str(1000 + id)

    event = MagicMock()
    event.id = id
    event.event_date = event_date
    event.event_root_code = event_root_code
    event.goldstein_scale = goldstein_scale
    event.num_mentions = num_mentions
    event.avg_tone = avg_tone
    event.actor1_code = actor1_code
    event.actor2_code = actor2_code
    event.actor1_country_code = actor1_country_code
    event.actor2_country_code = actor2_country_code
    event.actor1_name = actor1_name
    event.actor2_name = actor2_name
    event.action_geo_country_code = action_geo_country_code
    event.action_geo_name = None
    event.global_event_id = global_event_id
    event.source_url = None
    event.action_geo_lat = 50.0
    event.action_geo_long = 30.0
    return event


def make_market_data(
    id: int = 1,
    symbol: str = "SPY",
    market_date: date = None,
    close: float = 450.0,
    log_return: float = 0.001,
    daily_return: float = 0.001,
    volume: int = 1000000,
):
    """Create a mock MarketData object."""
    if market_date is None:
        market_date = date.today() - timedelta(days=5)

    md = MagicMock()
    md.id = id
    md.symbol = symbol
    md.date = market_date
    md.close = Decimal(str(close))
    md.log_return = log_return
    md.daily_return = daily_return
    md.volume = volume
    md.open = Decimal(str(close - 1))
    md.high = Decimal(str(close + 1))
    md.low = Decimal(str(close - 2))
    md.adj_close = Decimal(str(close))
    return md


def make_event_series(
    n: int = 30,
    start_date: date = None,
    base_goldstein: float = -2.0,
    noise: float = 3.0,
):
    """Create a series of mock events over n days."""
    if start_date is None:
        start_date = date.today() - timedelta(days=n)

    events = []
    for i in range(n):
        d = start_date + timedelta(days=i)
        goldstein = base_goldstein + np.random.uniform(-noise, noise)
        code = np.random.choice(["03", "04", "14", "17", "18", "19", "20"])
        events.append(make_event(
            id=i + 1,
            event_date=d,
            event_root_code=code,
            goldstein_scale=goldstein,
            num_mentions=np.random.randint(5, 100),
            avg_tone=goldstein / 2,
        ))
    return events


def make_market_series(
    n: int = 30,
    symbol: str = "SPY",
    start_date: date = None,
    base_price: float = 450.0,
    volatility: float = 0.01,
):
    """Create a series of mock market data over n trading days."""
    if start_date is None:
        start_date = date.today() - timedelta(days=n)

    data = []
    price = base_price
    for i in range(n):
        d = start_date + timedelta(days=i)
        ret = np.random.normal(0, volatility)
        price *= (1 + ret)
        data.append(make_market_data(
            id=i + 1,
            symbol=symbol,
            market_date=d,
            close=price,
            log_return=ret,
            daily_return=ret,
            volume=int(np.random.uniform(500000, 2000000)),
        ))
    return data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_events():
    """30 days of sample events."""
    return make_event_series(30)


@pytest.fixture
def sample_market_data():
    """30 days of sample market data for SPY."""
    return make_market_series(30, "SPY")


@pytest.fixture
def sample_date_range():
    """Standard 30-day date range."""
    end = date.today()
    start = end - timedelta(days=30)
    return start, end


@pytest.fixture
def mock_get_session():
    """Mock the database session context manager."""
    with patch("src.db.connection.get_session") as mock:
        session = MagicMock()
        mock.return_value.__enter__ = MagicMock(return_value=session)
        mock.return_value.__exit__ = MagicMock(return_value=False)
        yield session


@pytest.fixture
def feature_df():
    """Sample DataFrame matching feature engineering output."""
    n = 50
    dates = [date.today() - timedelta(days=n - i) for i in range(n)]
    np.random.seed(42)

    return pd.DataFrame({
        "date": dates,
        "close": np.cumsum(np.random.normal(0, 1, n)) + 450,
        "log_return": np.random.normal(0, 0.01, n),
        "daily_return": np.random.normal(0, 0.01, n),
        "abs_return": np.abs(np.random.normal(0, 0.01, n)),
        "rolling_mean": np.random.normal(0, 0.001, n),
        "rolling_std": np.abs(np.random.normal(0.01, 0.002, n)),
        "z_score": np.random.normal(0, 1, n),
        "volume": np.random.randint(500000, 2000000, n),
        "volume_change": np.random.normal(0, 0.1, n),
        "volume_zscore": np.random.normal(0, 1, n),
        "goldstein_mean": np.random.normal(-1, 3, n),
        "goldstein_min": np.random.normal(-5, 2, n),
        "event_count": np.random.randint(0, 20, n),
        "mentions_total": np.random.randint(0, 500, n),
    })
