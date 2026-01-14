# Geopolitical Market Tracker

A data science portfolio project that ingests global geopolitical events and correlates them with financial market movements to surface non-obvious patterns, detect anomalies, and provide actionable insights.

## About This Project

This is a personal learning project designed to develop and demonstrate a wide range of data science and engineering skills. The goal is to become a well-rounded data scientist by building a complete, production-ready system from scratch.

**Skills practiced in this project:**
- **Data Engineering**: ETL pipelines, database design, API development
- **Statistical Analysis**: Event studies, regression, anomaly detection
- **Machine Learning**: Classification models, feature engineering
- **Software Engineering**: Clean architecture, testing, Docker containerization
- **DevOps**: CI/CD, orchestration with Prefect, Makefile automation
- **Visualization**: Interactive dashboards with Streamlit and Plotly

**Built with Claude Code** - This project leverages [Claude Code](https://claude.com/claude-code) as an AI pair programmer to accelerate development, learn best practices, and explore new technologies efficiently.

## Overview

This system answers questions like:
- When a specific type of event occurs (military action, sanctions, protests), which assets typically react?
- How long does it take for markets to price in geopolitical events?
- Are there market moves happening without corresponding news events?
- Which regions/countries have the strongest event-to-market correlations?

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Prefect Orchestration                            │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐    │
│  │  Ingest Events   │ → │  Ingest Market   │ → │   Run Analysis   │    │
│  │    (GDELT)       │   │ (Yahoo Finance)  │   │                  │    │
│  └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘    │
└───────────┼──────────────────────┼──────────────────────┼──────────────┘
            │                      │                      │
            ▼                      ▼                      │
   ┌─────────────────┐    ┌─────────────────┐             │
   │     GDELT       │    │  Yahoo Finance  │             │
   │   (HTTP API)    │    │   (yfinance)    │             │
   └─────────────────┘    └─────────────────┘             │
                                                          │
            ┌─────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────┐
│       PostgreSQL          │
│    (SQLAlchemy ORM)       │
│  ┌───────┐ ┌───────────┐  │
│  │Events │ │MarketData │  │
│  └───────┘ └───────────┘  │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│     Analysis Layer        │
│  - Event Studies (CAR)    │
│  - Anomaly Detection      │
│  - Regression             │
│  - Classification         │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│    Presentation Layer     │
│  - FastAPI REST API       │
│  - Streamlit Dashboard    │
└───────────────────────────┘
```

## Quick Start

### Option 1: Docker (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/geopolitical-market-tracker.git
cd geopolitical-market-tracker

# 2. Start all services (database, API, dashboard)
make up

# 3. Wait for services to be healthy, then ingest data
make ingest-all

# 4. Access the applications
#    Dashboard: http://localhost:8501
#    API Docs:  http://localhost:8000/docs
#    Database:  localhost:5432
```

To stop the services:
```bash
make down
```

### Option 2: Local Development

```bash
# 1. Clone and setup virtual environment
git clone https://github.com/YOUR_USERNAME/geopolitical-market-tracker.git
cd geopolitical-market-tracker
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Start database only (requires Docker)
make up-db

# 3. Run database migrations
alembic upgrade head

# 4. Ingest data (fetches last 7 days of events + 30 days of market data)
make ingest-all

# 5. Run the dashboard
make dev-dashboard

# 6. (Optional) Run the API in a separate terminal
make dev-api
```

### Verify It's Working

After completing either option, you should be able to:

1. **View the Dashboard** at http://localhost:8501 - see event counts and market data
2. **Explore the API** at http://localhost:8000/docs - interactive Swagger documentation
3. **Query events** via API: `curl http://localhost:8000/api/events?limit=5`

## Makefile Commands

```bash
# Docker Commands
make up              # Start all services (database, API, dashboard)
make down            # Stop all services
make restart         # Restart all services
make logs            # View logs from all services
make build           # Rebuild Docker images
make clean           # Stop and remove volumes (WARNING: deletes data!)
make status          # Show status of all services

# Development Commands
make dev-api         # Run API locally with hot reload
make dev-dashboard   # Run dashboard locally (direct DB mode)
make dev-dashboard-api  # Run dashboard in API mode

# Data Ingestion
make ingest-events   # Ingest GDELT events (last 7 days)
make ingest-market   # Ingest market data (last 30 days)
make ingest-all      # Ingest both events and market data

# Testing & Quality
make test            # Run tests
make lint            # Check code style (black, flake8)
make format          # Format code with black

# Setup
make install         # Install Python dependencies
make setup           # Full setup: install deps, start DB, run migrations

# Prefect Orchestration
make prefect-server  # Start Prefect server (UI at http://localhost:4200)
make prefect-worker  # Start worker to execute scheduled flows
make pipeline        # Run the full daily pipeline manually
make pipeline-deploy # Create scheduled deployments
make pipeline-events # Run only event ingestion
make pipeline-market # Run only market data ingestion
make pipeline-analysis # Run only analysis
```

## Project Structure

```
geopolitical-market-tracker/
├── src/
│   ├── ingestion/          # Data fetching
│   │   ├── gdelt.py        # GDELT event ingestion
│   │   └── market_data.py  # Yahoo Finance market data
│   ├── analysis/           # Statistical analysis
│   │   ├── event_study.py  # Cumulative Abnormal Returns (CAR)
│   │   ├── anomaly.py      # Unexplained moves, muted responses
│   │   ├── regression.py   # OLS regression analysis
│   │   └── classifier.py   # Market direction prediction
│   ├── db/                 # Database layer
│   │   ├── models.py       # SQLAlchemy models
│   │   ├── connection.py   # Session management
│   │   └── queries.py      # Reusable query functions
│   ├── api/                # FastAPI REST API
│   │   ├── main.py         # Application entry point
│   │   ├── schemas.py      # Pydantic models
│   │   └── routes/         # API endpoints
│   │       ├── events.py   # /api/events
│   │       ├── market.py   # /api/market
│   │       └── analysis.py # /api/analysis
│   └── config/             # Configuration
│       └── constants.py    # CAMEO codes, symbols, event groups
├── dashboard/              # Streamlit dashboard
│   ├── app.py              # Main entry point
│   ├── api_client.py       # HTTP client for API mode
│   └── views/              # Dashboard pages
│       ├── home.py         # Overview metrics and charts
│       ├── event_map.py    # Interactive Folium map
│       ├── market_analysis.py  # Event study results
│       ├── anomalies.py    # Anomaly detection results
│       └── predictions.py  # Classification model demo
├── flows/                  # Prefect orchestration
│   ├── daily_pipeline.py   # Main orchestration flow
│   ├── ingest_events.py    # GDELT ingestion flow
│   ├── ingest_market.py    # Market data ingestion flow
│   └── run_analysis.py     # Analysis flow
├── notebooks/              # Jupyter notebooks
│   └── 01_data_exploration.ipynb
├── alembic/                # Database migrations
├── scripts/                # Utility scripts
├── Dockerfile.api          # API container
├── Dockerfile.dashboard    # Dashboard container
├── docker-compose.yml      # Full stack orchestration
├── Makefile                # DevOps commands
└── requirements.txt        # Python dependencies
```

## Features

### Data Ingestion

- **GDELT Events**: Fetches geopolitical events from GDELT's daily CSV exports
  - Filters for significant events (5+ media mentions)
  - Maps CAMEO event codes to human-readable categories
  - Extracts geographic coordinates for mapping

- **Market Data**: Fetches OHLCV data via Yahoo Finance
  - 33 tracked symbols across commodities, currencies, ETFs, and bonds
  - Calculates daily and log returns
  - Upserts to avoid duplicates

### Analysis Methods

| Method | Description | Output |
|--------|-------------|--------|
| **Event Study** | Measures abnormal returns around geopolitical events | CAR, t-statistic, p-value |
| **Anomaly Detection** | Finds unexplained market moves and muted responses | Z-scores, anomaly flags |
| **Regression** | Analyzes event-market relationships with OLS | Coefficients, R², p-values |
| **Classification** | Predicts market direction (UP/DOWN) | Probability, feature importance |

### REST API

The FastAPI provides programmatic access to all data and analysis:

```bash
# Get events
GET /api/events?start_date=2024-01-01&end_date=2024-01-31&country_code=US

# Get market data
GET /api/market/CL=F?start_date=2024-01-01&end_date=2024-01-31

# Get analysis summary
GET /api/analysis/summary

# Make predictions
POST /api/analysis/predict
{
  "symbol": "CL=F",
  "goldstein_scale": -5.0,
  "num_mentions": 50,
  "event_root_code": "19"
}
```

Full API documentation available at `http://localhost:8000/docs` when running.

### Dashboard

The Streamlit dashboard provides 5 interactive pages:

1. **Home**: Key metrics, event counts by type/country, recent events table
2. **Event Map**: Interactive Folium map with color-coded event markers
3. **Market Analysis**: Performance summary, event study results, price charts
4. **Anomalies**: Unexplained moves, muted responses, Z-score analysis
5. **Predictions**: Classification model demo with feature importance

### Dual-Mode Architecture

The dashboard supports two data access modes:

- **Direct DB Mode** (default): Connects directly to PostgreSQL for local development
- **API Mode**: Communicates via REST API for containerized deployment

Set via environment variables:
```bash
USE_API=true API_URL=http://api:8000 streamlit run dashboard/app.py
```

### Prefect Orchestration

The project uses [Prefect](https://www.prefect.io/) for workflow orchestration. Prefect provides:
- **Scheduled execution**: Run pipelines daily, weekly, or on custom schedules
- **Retries**: Automatic retry on failure with configurable delays
- **Caching**: Avoid re-processing data that's already been fetched
- **Observability**: Web UI for monitoring flow runs and debugging

#### Available Flows

| Flow | Description | Schedule |
|------|-------------|----------|
| `daily-pipeline` | Full pipeline: ingest → analyze | Daily 8 PM UTC |
| `ingest-gdelt-events` | Fetch GDELT events | Daily 6 AM UTC |
| `ingest-market-data` | Fetch Yahoo Finance data | Daily 7 PM UTC (Mon-Fri) |
| `run-analysis` | Event studies, anomaly detection | Daily 8 PM UTC |

#### Quick Start with Prefect

**Option 1: Run locally (for development)**
```bash
# Run pipeline once (no scheduling)
make pipeline

# Or set up scheduled runs locally
make prefect-server    # Terminal 1
make pipeline-deploy   # Terminal 2
make prefect-worker    # Terminal 2
```

**Option 2: Run in Docker (for production)**
```bash
# Start Prefect server + worker in containers
make up-prefect

# Create scheduled deployments
make prefect-deploy-docker

# Or run pipeline once in Docker
make pipeline-docker

# Start everything (app + Prefect)
make up-all
```

View the Prefect UI at http://localhost:4200

#### How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    Daily Pipeline                        │
│                                                         │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   │
│  │   Ingest    │   │   Ingest    │   │    Run      │   │
│  │   GDELT     │ → │   Market    │ → │  Analysis   │   │
│  │   Events    │   │    Data     │   │             │   │
│  └─────────────┘   └─────────────┘   └─────────────┘   │
│                                                         │
│  @task: fetch      @task: fetch      @task: event_study │
│  @task: store      @task: store      @task: anomalies   │
│                                      @task: report      │
└─────────────────────────────────────────────────────────┘
```

Each flow is built from **tasks** (individual units of work) that:
- Retry automatically on failure
- Cache results to avoid redundant work
- Log progress for debugging

## Data Sources

- **[GDELT](https://www.gdeltproject.org/)**: Global Database of Events, Language, and Tone - real-time monitoring of global news
- **[Yahoo Finance](https://finance.yahoo.com/)**: Daily OHLCV data for stocks, ETFs, commodities, currencies

### Tracked Symbols

| Category | Symbols |
|----------|---------|
| Commodities | CL=F (Oil), GC=F (Gold), SI=F (Silver), NG=F (Natural Gas), HG=F (Copper) |
| Currencies | EURUSD=X, GBPUSD=X, USDJPY=X, USDCHF=X, AUDUSD=X, USDCAD=X |
| Indices | ^VIX, ^GSPC (S&P 500), ^DJI (Dow), ^IXIC (Nasdaq) |
| ETFs | SPY, QQQ, EFA, EEM, XLE, XLF, GLD, USO, UNG |
| Bonds | ^TNX (10Y), ^TYX (30Y), TLT, IEF, SHY |

## Docker Services

| Service | Port | Description |
|---------|------|-------------|
| `db` | 5432 | PostgreSQL 15 database |
| `api` | 8000 | FastAPI REST API |
| `dashboard` | 8501 | Streamlit dashboard |
| `prefect-server` | 4200 | Prefect orchestration UI |
| `prefect-worker` | - | Executes scheduled flows |

**Docker Profiles:**
- Default (`make up`): db, api, dashboard
- Prefect (`make up-prefect`): prefect-server, prefect-worker
- All (`make up-all`): Everything

Health checks ensure services start in the correct order: db → api → dashboard

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | postgresql://postgres:postgres@localhost:5432/geopolitical_tracker | Database connection string |
| `USE_API` | false | Dashboard: use API instead of direct DB |
| `API_URL` | http://localhost:8000 | API base URL when USE_API=true |

## License

MIT
