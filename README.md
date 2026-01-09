# Geopolitical Market Tracker

A data science portfolio project that ingests global geopolitical events and correlates them with financial market movements to surface non-obvious patterns, detect anomalies, and provide actionable insights.

## Overview

This system answers questions like:
- When a specific type of event occurs (military action, sanctions, protests), which assets typically react?
- How long does it take for markets to price in geopolitical events?
- Are there market moves happening without corresponding news events?
- Which regions/countries have the strongest event-to-market correlations?

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     GDELT       │    │  Yahoo Finance  │    │      FRED       │
│  (events feed)  │    │  (market data)  │    │ (macro data)    │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Ingestion Layer     │
                    │   (Prefect flows)     │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │     PostgreSQL        │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Analysis Layer      │
                    │ (Event studies, CAR)  │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   FastAPI + Streamlit │
                    └───────────────────────┘
```

## Setup

### Prerequisites
- Python 3.10+
- Docker & Docker Compose

### Quick Start

1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/geopolitical-market-tracker.git
cd geopolitical-market-tracker
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Create `.env` file
```bash
cat <<EOF > .env
# Database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=geopolitical_tracker
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/geopolitical_tracker

# Prefect
PREFECT_API_URL=http://localhost:4200/api
EOF
```

4. Start the database
```bash
docker compose up -d db
```

5. Run migrations
```bash
alembic upgrade head
```

6. Verify database connection
```bash
python scripts/setup_db.py
```

7. Ingest data
```bash
python -c "
from datetime import date, timedelta
from src.ingestion.gdelt import GDELTIngestion
from src.ingestion.market_data import MarketDataIngestion

# Ingest GDELT events (last 7 days)
GDELTIngestion().ingest_date_range(date.today() - timedelta(days=7), date.today() - timedelta(days=1))

# Ingest market data (last 30 days)
MarketDataIngestion().ingest_all_symbols(date.today() - timedelta(days=30), date.today())
"
```

## Project Structure

```
geopolitical-market-tracker/
├── src/
│   ├── ingestion/      # Data fetching (GDELT, market data)
│   ├── analysis/       # Correlation, anomaly detection
│   ├── db/             # Database models and queries
│   ├── api/            # FastAPI endpoints
│   └── config/         # Settings and constants
├── flows/              # Prefect orchestration flows
├── dashboard/          # Streamlit visualization
├── notebooks/          # Exploration and prototyping
├── tests/              # Unit and integration tests
├── scripts/            # Database setup, backfill scripts
└── docs/               # Documentation
```

## Data Sources

- **GDELT**: Global Database of Events, Language, and Tone
- **Yahoo Finance**: Daily OHLCV data for stocks, ETFs, commodities, currencies
- **FRED**: Federal Reserve Economic Data (macroeconomic indicators)

## Key Features

- **Event Study Analysis**: Calculate cumulative abnormal returns (CAR) with statistical significance testing
- **Anomaly Detection**: Identify events that should move markets but don't, and vice versa
- **Pattern Recognition**: Learn historical event-market correlations
- **REST API**: Query events, correlations, and anomalies programmatically
- **Dashboard**: Visualize events on a map with market overlays

## License

MIT
