"""
FastAPI Application Entry Point.

USAGE:
------
    # Development
    uvicorn src.api.main:app --reload

    # Production
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

DOCUMENTATION:
--------------
    - Swagger UI: http://localhost:8000/docs
    - ReDoc: http://localhost:8000/redoc
    - OpenAPI JSON: http://localhost:8000/openapi.json
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time

from sqlalchemy import text

from src.api.routes import (
    events_router,
    market_router,
    analysis_router,
    correlation_router,
    patterns_router,
    predictions_router,
)
from src.api.schemas import HealthResponse, ErrorResponse
from src.db.connection import get_session


# =============================================================================
# APP LIFECYCLE
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.

    This runs code on startup and shutdown.
    Good for initializing database connections, loading models, etc.
    """
    # Startup
    print("🚀 Starting Geopolitical Market Tracker API...")

    # Verify database connection
    try:
        with get_session() as session:
            session.execute(text("SELECT 1"))
        print("✓ Database connection verified")
    except Exception as e:
        print(f"⚠ Database connection failed: {e}")

    yield  # App runs here

    # Shutdown
    print("👋 Shutting down API...")


# =============================================================================
# APP CONFIGURATION
# =============================================================================

app = FastAPI(
    title="Geopolitical Market Tracker API",
    description="""
    REST API for querying geopolitical events and their market impact.

    ## Features

    - **Events**: Query GDELT geopolitical events with filters
    - **Market Data**: Access financial market prices and returns
    - **Analysis**: View event study results and detected anomalies
    - **Predictions**: Predict market direction based on event characteristics

    ## Data Sources

    - **GDELT**: Global Database of Events, Language, and Tone
    - **Yahoo Finance**: Daily OHLCV data for stocks, ETFs, commodities

    ## Authentication

    Currently no authentication required (development mode).
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# =============================================================================
# MIDDLEWARE
# =============================================================================

# CORS - Allow cross-origin requests (needed for web frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add X-Response-Time header to all responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Response-Time"] = f"{process_time:.4f}s"
    return response


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions gracefully."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc) if app.debug else "An unexpected error occurred",
            "status_code": 500,
        },
    )


# =============================================================================
# ROUTERS
# =============================================================================

# Mount routers under /api prefix
app.include_router(events_router, prefix="/api")
app.include_router(market_router, prefix="/api")
app.include_router(analysis_router, prefix="/api")
app.include_router(correlation_router, prefix="/api")
app.include_router(patterns_router, prefix="/api")
app.include_router(predictions_router, prefix="/api")


# =============================================================================
# ROOT ENDPOINTS
# =============================================================================

@app.get("/", tags=["Root"])
def root():
    """
    Root endpoint - API information.

    Returns basic info and links to documentation.
    """
    return {
        "name": "Geopolitical Market Tracker API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "events": "/api/events",
            "market": "/api/market",
            "analysis": "/api/analysis",
            "correlation": "/api/correlation",
            "patterns": "/api/patterns",
            "predictions": "/api/predictions",
            "health": "/health",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Health check endpoint.

    Used by load balancers and monitoring tools to verify the API is running.
    """
    # Check database
    db_status = "connected"
    try:
        with get_session() as session:
            session.execute(text("SELECT 1"))
    except Exception:
        db_status = "disconnected"

    return HealthResponse(
        status="healthy" if db_status == "connected" else "degraded",
        database=db_status,
        version="1.0.0",
    )


# =============================================================================
# DEVELOPMENT SERVER
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,  # Auto-reload on code changes
    )
