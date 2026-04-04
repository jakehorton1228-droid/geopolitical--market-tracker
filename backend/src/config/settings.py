"""Application configuration settings."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths — points to backend/ directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load environment variables from .env file
# Search backend/ first, then repo root (parent of backend/)
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT.parent / ".env")
DATA_DIR = PROJECT_ROOT / "data"

# Database
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://gmt:gmt_password@localhost:5432/geopolitical_market_tracker"
)

# API Keys
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

# GDELT settings
GDELT_EVENTS_URL = "http://data.gdeltproject.org/events"
GDELT_MIN_MENTIONS = 200  # Minimum mentions to consider an event significant

# RSS Feed settings
# Each feed is a dict with a name (our internal label) and URL.
# These are the "world news" feeds — focused on geopolitics, not sports/entertainment.
RSS_FEEDS = [
    # Google News RSS aggregates Reuters world coverage (Reuters killed their direct feed)
    {"name": "reuters", "url": "https://news.google.com/rss/search?q=site:reuters.com+world&hl=en-US&gl=US&ceid=US:en"},
    # Google News RSS aggregates AP world coverage (AP has no reliable public feed)
    {"name": "ap", "url": "https://news.google.com/rss/search?q=site:apnews.com+world&hl=en-US&gl=US&ceid=US:en"},
    {"name": "bbc", "url": "https://feeds.bbci.co.uk/news/world/rss.xml"},
    {"name": "aljazeera", "url": "https://www.aljazeera.com/xml/rss/all.xml"},
]
RSS_REQUEST_TIMEOUT = 30  # Seconds to wait for a feed response

# FRED (Federal Reserve Economic Data) settings
# Each series has an ID (FRED's identifier), a human-readable name, and frequency.
# Frequency is informational — all observations are stored the same way.
FRED_SERIES = [
    {"id": "GDP", "name": "Gross Domestic Product", "frequency": "quarterly"},
    {"id": "CPIAUCSL", "name": "Consumer Price Index", "frequency": "monthly"},
    {"id": "UNRATE", "name": "Unemployment Rate", "frequency": "monthly"},
    {"id": "DFF", "name": "Federal Funds Rate", "frequency": "daily"},
    {"id": "DGS10", "name": "10-Year Treasury Yield", "frequency": "daily"},
    {"id": "UMCSENT", "name": "Consumer Sentiment (Michigan)", "frequency": "monthly"},
]
FRED_REQUEST_TIMEOUT = 30  # Seconds to wait for FRED API response

# Polymarket prediction market settings
# No API key needed — Polymarket's Gamma API is fully public.
# We filter events by tags to find geopolitically relevant markets.
POLYMARKET_GEOPOLITICAL_TAGS = {
    "geopolitics", "world", "politics", "economy", "economic-policy",
    "elections", "global-elections", "world-elections",
    "foreign-policy", "diplomacy-ceasefire", "military-strikes",
    "iran", "russia", "china", "ukraine", "israel", "middle-east",
    "fed", "fed-rates", "tariffs", "sanctions", "oil", "commodities",
}
POLYMARKET_EXCLUDE_TAGS = {
    "sports", "soccer", "nba", "nfl", "nhl", "mlb", "basketball", "baseball",
    "football", "mma", "games", "epl", "crypto", "pop-culture", "tweets-markets",
}
POLYMARKET_REQUEST_TIMEOUT = 30

# LLM (Ollama — local inference)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_MAX_TOKENS = 4096

# LangSmith (LLM observability)
# These env vars are read automatically by LangGraph/LangChain:
#   LANGCHAIN_TRACING_V2=true  — enables tracing
#   LANGCHAIN_API_KEY=...      — LangSmith API key
#   LANGCHAIN_PROJECT=...      — project name in LangSmith dashboard
LANGSMITH_TRACING_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "gmip-intelligence-pipeline")

# MLflow (experiment tracking)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "gmip-event-impact")

# Analysis settings
EVENT_STUDY_ESTIMATION_WINDOW = 30  # Days for baseline calculation
EVENT_STUDY_EVENT_WINDOW_BEFORE = 1  # Days before event
EVENT_STUDY_EVENT_WINDOW_AFTER = 5  # Days after event
SIGNIFICANCE_LEVEL = 0.05  # p-value threshold

# Anomaly detection
ANOMALY_GOLDSTEIN_THRESHOLD = 5.0  # Min absolute Goldstein for "big event"
ANOMALY_MOVE_THRESHOLD_PCT = 1.0  # Expected minimum market reaction
ANOMALY_ZSCORE_THRESHOLD = 2.0  # Std devs for "unexplained move"
