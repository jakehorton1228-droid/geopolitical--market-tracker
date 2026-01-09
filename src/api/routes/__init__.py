"""API route modules."""

from src.api.routes.events import router as events_router
from src.api.routes.market import router as market_router
from src.api.routes.analysis import router as analysis_router

__all__ = ["events_router", "market_router", "analysis_router"]
