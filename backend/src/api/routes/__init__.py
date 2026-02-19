"""API route modules."""

from src.api.routes.events import router as events_router
from src.api.routes.market import router as market_router
from src.api.routes.analysis import router as analysis_router
from src.api.routes.correlation import router as correlation_router
from src.api.routes.patterns import router as patterns_router
from src.api.routes.predictions import router as predictions_router
from src.api.routes.agent import router as agent_router

__all__ = [
    "events_router",
    "market_router",
    "analysis_router",
    "correlation_router",
    "patterns_router",
    "predictions_router",
    "agent_router",
]
