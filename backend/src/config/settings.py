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

# AI Agent
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
AGENT_MODEL = "claude-sonnet-4-20250514"
AGENT_MAX_TOKENS = 4096

# Analysis settings
EVENT_STUDY_ESTIMATION_WINDOW = 30  # Days for baseline calculation
EVENT_STUDY_EVENT_WINDOW_BEFORE = 1  # Days before event
EVENT_STUDY_EVENT_WINDOW_AFTER = 5  # Days after event
SIGNIFICANCE_LEVEL = 0.05  # p-value threshold

# Anomaly detection
ANOMALY_GOLDSTEIN_THRESHOLD = 5.0  # Min absolute Goldstein for "big event"
ANOMALY_MOVE_THRESHOLD_PCT = 1.0  # Expected minimum market reaction
ANOMALY_ZSCORE_THRESHOLD = 2.0  # Std devs for "unexplained move"
