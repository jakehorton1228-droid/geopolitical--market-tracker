"""API route modules."""

from src.api.routes.events import router as events_router
from src.api.routes.market import router as market_router
from src.api.routes.analysis import router as analysis_router
from src.api.routes.correlation import router as correlation_router
from src.api.routes.patterns import router as patterns_router
from src.api.routes.predictions import router as predictions_router
from src.api.routes.indicators import router as indicators_router
from src.api.routes.headlines import router as headlines_router
from src.api.routes.prediction_markets import router as prediction_markets_router
from src.api.routes.briefing import router as briefing_router

__all__ = [
    "events_router",
    "market_router",
    "analysis_router",
    "correlation_router",
    "patterns_router",
    "predictions_router",
    "indicators_router",
    "headlines_router",
    "prediction_markets_router",
    "briefing_router",
]
