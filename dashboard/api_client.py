"""
API Client for Dashboard.

This module provides a clean interface for the dashboard to communicate
with the FastAPI backend. In containerized deployment, the dashboard
uses this instead of direct database access.

USAGE:
------
    from dashboard.api_client import APIClient

    client = APIClient()  # Uses API_URL from environment

    # Get events
    events = client.get_events(country_code="RUS", limit=50)

    # Get market data
    prices = client.get_market_data(symbol="CL=F")

    # Make prediction
    prediction = client.predict(symbol="CL=F", goldstein=-5.0)

ENVIRONMENT:
------------
    API_URL: Base URL of the FastAPI service (default: http://localhost:8000)
"""

import os
from datetime import date
from typing import Any
import requests
from requests.exceptions import RequestException


class APIClient:
    """
    HTTP client for communicating with the FastAPI backend.

    This abstracts away the HTTP details and provides a clean Python interface.
    """

    def __init__(self, base_url: str | None = None):
        """
        Initialize the API client.

        Args:
            base_url: API base URL. Defaults to API_URL env var or localhost.
        """
        self.base_url = base_url or os.getenv("API_URL", "http://localhost:8000")
        self.timeout = 30  # seconds

    def _request(self, method: str, endpoint: str, **kwargs) -> dict | list | None:
        """Make HTTP request to the API."""
        url = f"{self.base_url}{endpoint}"

        try:
            response = requests.request(
                method,
                url,
                timeout=self.timeout,
                **kwargs,
            )
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            print(f"API request failed: {e}")
            return None

    def _get(self, endpoint: str, params: dict | None = None) -> dict | list | None:
        """Make GET request."""
        return self._request("GET", endpoint, params=params)

    def _post(self, endpoint: str, json: dict | None = None) -> dict | None:
        """Make POST request."""
        return self._request("POST", endpoint, json=json)

    # =========================================================================
    # HEALTH
    # =========================================================================

    def health_check(self) -> dict | None:
        """Check if API is healthy."""
        return self._get("/health")

    def is_healthy(self) -> bool:
        """Return True if API is reachable and healthy."""
        result = self.health_check()
        return result is not None and result.get("status") == "healthy"

    # =========================================================================
    # EVENTS
    # =========================================================================

    def get_events(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        country_code: str | None = None,
        event_group: str | None = None,
        min_mentions: int | None = None,
        min_goldstein: float | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """
        Get events with optional filters.

        Returns list of event dictionaries.
        """
        params = {"limit": limit, "offset": offset}

        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()
        if country_code:
            params["country_code"] = country_code
        if event_group:
            params["event_group"] = event_group
        if min_mentions:
            params["min_mentions"] = min_mentions
        if min_goldstein:
            params["min_goldstein"] = min_goldstein

        result = self._get("/api/events", params=params)
        return result if result else []

    def get_event(self, event_id: int) -> dict | None:
        """Get single event by ID."""
        return self._get(f"/api/events/{event_id}")

    def get_events_count(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> int:
        """Get total event count."""
        params = {}
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()

        result = self._get("/api/events/count", params=params)
        return result.get("count", 0) if result else 0

    def get_events_by_country(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Get event counts grouped by country."""
        params = {"limit": limit}
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()

        result = self._get("/api/events/by-country", params=params)
        return result if result else []

    def get_events_by_type(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[dict]:
        """Get event counts grouped by type."""
        params = {}
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()

        result = self._get("/api/events/by-type", params=params)
        return result if result else []

    # =========================================================================
    # MARKET DATA
    # =========================================================================

    def get_market_data(
        self,
        symbol: str | None = None,
        symbols: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Get market data with optional filters."""
        params = {"limit": limit}

        if symbol:
            params["symbol"] = symbol
        if symbols:
            params["symbols"] = ",".join(symbols)
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()

        result = self._get("/api/market", params=params)
        return result if result else []

    def get_symbols(self) -> dict:
        """Get all available market symbols."""
        result = self._get("/api/market/symbols")
        return result if result else {"symbols": {}, "total": 0}

    def get_symbol_data(
        self,
        symbol: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[dict]:
        """Get data for a specific symbol."""
        params = {}
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()

        result = self._get(f"/api/market/{symbol}", params=params)
        return result if result else []

    def get_latest_price(self, symbol: str) -> dict | None:
        """Get latest price for a symbol."""
        return self._get(f"/api/market/{symbol}/latest")

    def get_symbol_stats(
        self,
        symbol: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> dict | None:
        """Get statistics for a symbol."""
        params = {}
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()

        return self._get(f"/api/market/{symbol}/stats", params=params)

    # =========================================================================
    # ANALYSIS
    # =========================================================================

    def get_analysis_results(
        self,
        event_id: int | None = None,
        symbol: str | None = None,
        is_significant: bool | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get analysis results."""
        params = {"limit": limit}
        if event_id:
            params["event_id"] = event_id
        if symbol:
            params["symbol"] = symbol
        if is_significant is not None:
            params["is_significant"] = is_significant

        result = self._get("/api/analysis/results", params=params)
        return result if result else []

    def get_anomalies(
        self,
        anomaly_type: str | None = None,
        symbol: str | None = None,
        min_score: float | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get detected anomalies."""
        params = {"limit": limit}
        if anomaly_type:
            params["anomaly_type"] = anomaly_type
        if symbol:
            params["symbol"] = symbol
        if min_score:
            params["min_score"] = min_score

        result = self._get("/api/analysis/anomalies", params=params)
        return result if result else []

    def get_significant_results(
        self,
        symbol: str | None = None,
        min_car: float | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Get significant event study results."""
        params = {"limit": limit}
        if symbol:
            params["symbol"] = symbol
        if min_car:
            params["min_car"] = min_car

        result = self._get("/api/analysis/significant", params=params)
        return result if result else []

    def get_analysis_summary(self) -> dict:
        """Get analysis summary statistics."""
        result = self._get("/api/analysis/summary")
        return result if result else {}

    def predict(
        self,
        symbol: str,
        goldstein_scale: float,
        num_mentions: int = 1,
        avg_tone: float = 0,
        is_violent_conflict: bool = False,
    ) -> dict | None:
        """
        Predict market direction based on event characteristics.

        Returns prediction with probability and confidence.
        """
        payload = {
            "symbol": symbol,
            "goldstein_scale": goldstein_scale,
            "num_mentions": num_mentions,
            "avg_tone": avg_tone,
            "is_violent_conflict": is_violent_conflict,
        }
        return self._post("/api/analysis/predict", json=payload)


# Singleton instance for convenience
_client: APIClient | None = None


def get_client() -> APIClient:
    """Get or create the singleton API client."""
    global _client
    if _client is None:
        _client = APIClient()
    return _client
