"""
FastAPI REST API Module.

USAGE:
------
    # Start the API server
    uvicorn src.api.main:app --reload

    # Or run directly
    python -m src.api.main

ENDPOINTS:
----------
    GET  /                     - API info
    GET  /health               - Health check
    GET  /docs                 - Swagger UI
    GET  /redoc                - ReDoc

    GET  /api/events           - List events
    GET  /api/events/{id}      - Get event
    GET  /api/events/by-country - Events by country
    GET  /api/events/by-type    - Events by type

    GET  /api/market           - List market data
    GET  /api/market/symbols   - Available symbols
    GET  /api/market/{symbol}  - Symbol data
    GET  /api/market/{symbol}/latest - Latest price
    GET  /api/market/{symbol}/stats  - Statistics

    GET  /api/analysis/results    - Analysis results
    GET  /api/analysis/anomalies  - Detected anomalies
    GET  /api/analysis/significant - Significant results
    POST /api/analysis/predict    - Predict direction
"""

from src.api.main import app

__all__ = ["app"]
