"""
FRED (Federal Reserve Economic Data) Ingestion Module.

WHAT IS FRED?
-------------
FRED is a free API from the Federal Reserve Bank of St. Louis, providing access
to 800,000+ economic time series. It's the standard source for macro data used
by quant desks, economists, and policy researchers.

WHY THESE INDICATORS?
---------------------
We pull six series that capture different facets of the macro environment:

1. GDP (quarterly)       — overall economic output, markets trend with growth
2. CPI (monthly)         — inflation, drives Fed policy and equity valuations
3. Unemployment (monthly)— labor market health, market-moving on release day
4. Fed Funds Rate (daily)— cost of money, affects everything
5. 10Y Treasury (daily)  — risk-free rate benchmark, inversely correlated with equities
6. Consumer Sentiment    — Michigan survey, our macro sentiment signal

Together these give us the macro context around geopolitical events. A military
conflict during a recession looks very different to markets than the same conflict
during a boom.

HOW THE API WORKS:
------------------
Simple REST: GET /fred/series/observations?series_id=GDP&api_key=KEY&file_type=json
Returns JSON: {"observations": [{"date": "2024-01-01", "value": "27610.1"}, ...]}

Values come as strings (including "." for missing data), so we use safe_float()
to handle conversion defensively.

USAGE:
------
    from src.ingestion.fred import FREDIngestion

    fred = FREDIngestion()

    # Fetch without storing
    observations = fred.fetch_series("GDP")

    # Fetch and store all configured series
    counts = fred.ingest_all_series()
    print(counts)  # {"GDP": 4, "CPIAUCSL": 12, ...}
"""

import logging
from datetime import date, timedelta

import requests

from src.config.settings import FRED_API_KEY, FRED_SERIES, FRED_REQUEST_TIMEOUT
from src.db.connection import get_session
from src.db.queries import upsert_economic_indicator

logger = logging.getLogger(__name__)

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


def safe_float(value: str | None) -> float | None:
    """
    Safely convert FRED value strings to floats.

    FRED returns values as strings, including "." for missing/unreported data.
    This mirrors the safe_float pattern used in GDELT ingestion.
    """
    if value is None or value.strip() == "" or value.strip() == ".":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


class FREDIngestion:
    """
    Handles fetching and storing economic indicators from the FRED API.

    Follows the same pattern as RSSIngestion / GDELTIngestion:
    1. fetch_* methods retrieve data without storing
    2. ingest_* methods retrieve and store in the database
    3. Deduplication via (series_id, date) uniqueness
    """

    def __init__(
        self,
        api_key: str | None = None,
        series: list[dict] | None = None,
        timeout: int | None = None,
    ):
        """
        Initialize the FRED ingestion handler.

        Args:
            api_key: FRED API key. Defaults to FRED_API_KEY from settings.
            series: List of series config dicts with "id", "name", "frequency" keys.
                    Defaults to FRED_SERIES from settings.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key or FRED_API_KEY
        if not self.api_key:
            raise ValueError(
                "FRED_API_KEY not set. Get a free key at "
                "https://fred.stlouisfed.org/docs/api/api_key.html "
                "and set it in your .env file."
            )
        self.series = series or FRED_SERIES
        self.timeout = timeout or FRED_REQUEST_TIMEOUT

    def fetch_series(
        self,
        series_id: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[dict]:
        """
        Fetch observations for a single FRED series.

        This is the read-only method — fetches data without storing it.

        Args:
            series_id: FRED series ID (e.g., "GDP", "CPIAUCSL")
            start_date: Earliest observation date. Defaults to 1 year ago.
            end_date: Latest observation date. Defaults to today.

        Returns:
            List of dicts with "series_id", "date", "value" keys.
            Only includes observations with valid numeric values.
        """
        if start_date is None:
            start_date = date.today() - timedelta(days=365)
        if end_date is None:
            end_date = date.today()

        logger.info(
            f"Fetching FRED series {series_id} "
            f"from {start_date} to {end_date}"
        )

        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date.isoformat(),
            "observation_end": end_date.isoformat(),
            "sort_order": "desc",
        }

        try:
            response = requests.get(
                FRED_BASE_URL, params=params, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch FRED series '{series_id}': {e}")
            return []

        raw_observations = data.get("observations", [])
        observations = []

        for obs in raw_observations:
            value = safe_float(obs.get("value"))
            if value is None:
                continue  # Skip missing data points

            observations.append({
                "series_id": series_id,
                "date": obs["date"],  # FRED returns "YYYY-MM-DD" strings
                "value": value,
            })

        logger.info(
            f"Parsed {len(observations)} valid observations "
            f"from {len(raw_observations)} total for {series_id}"
        )
        return observations

    def ingest_series(
        self,
        series_id: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> int:
        """
        Fetch a single FRED series and store in the database.

        Mirrors RSSIngestion.ingest_feed() — fetch + store + return count.

        Args:
            series_id: FRED series ID
            start_date: Earliest observation date
            end_date: Latest observation date

        Returns:
            Number of NEW observations stored (skips duplicates).
            -1 on error.
        """
        # Find the series config to get the human-readable name
        series_config = next(
            (s for s in self.series if s["id"] == series_id),
            None,
        )
        series_name = series_config["name"] if series_config else series_id

        try:
            observations = self.fetch_series(series_id, start_date, end_date)
        except Exception as e:
            logger.error(f"Error fetching FRED series '{series_id}': {e}")
            return -1

        if not observations:
            return 0

        new_count = 0
        with get_session() as session:
            for obs in observations:
                indicator_data = {
                    "series_id": obs["series_id"],
                    "series_name": series_name,
                    "date": date.fromisoformat(obs["date"]),
                    "value": obs["value"],
                }
                result = upsert_economic_indicator(session, indicator_data)
                if result is not None:
                    new_count += 1

        logger.info(
            f"Ingested {series_id} ({series_name}): {new_count} new observations "
            f"({len(observations) - new_count} duplicates skipped)"
        )
        return new_count

    def ingest_all_series(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> dict[str, int]:
        """
        Fetch and store all configured FRED series.

        Mirrors RSSIngestion.ingest_all_feeds() — loops through all series,
        catches per-series errors so one failure doesn't stop the rest.

        Args:
            start_date: Earliest observation date (defaults per fetch_series)
            end_date: Latest observation date

        Returns:
            Dict of {series_id: count} for each series. -1 indicates an error.
        """
        results = {}

        for series_config in self.series:
            series_id = series_config["id"]
            try:
                count = self.ingest_series(series_id, start_date, end_date)
                results[series_id] = count
            except Exception as e:
                logger.error(f"Error ingesting FRED series '{series_id}': {e}")
                results[series_id] = -1

        total = sum(c for c in results.values() if c > 0)
        logger.info(
            f"FRED ingestion complete: {total} new observations "
            f"across {len(results)} series"
        )
        return results


# Convenience function for quick testing (matches fetch_sample_headlines pattern)
def fetch_sample_indicators(
    series_id: str = "DFF", days_back: int = 30
) -> list[dict]:
    """
    Fetch a few sample observations for testing/exploration.

    Uses DFF (Fed Funds Rate) by default since it's daily and always has data.

    Args:
        series_id: FRED series ID to fetch
        days_back: How many days of data to fetch

    Returns:
        List of observation dicts
    """
    fred = FREDIngestion()
    start_date = date.today() - timedelta(days=days_back)
    return fred.fetch_series(series_id, start_date=start_date)
