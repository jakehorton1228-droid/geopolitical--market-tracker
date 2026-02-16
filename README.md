# Geopolitical Market Tracker

A full-stack application that correlates global geopolitical events with financial market movements. It surfaces historical patterns, computes correlations, and predicts market direction using interpretable models.

## About This Project

This is a personal learning project focused on building a complete full-stack system from scratch — React frontend, Python backend, PostgreSQL database.

**Skills practiced:**
- **Frontend**: React, Vite, Tailwind CSS, Recharts, React Query
- **Backend**: FastAPI, SQLAlchemy, Pydantic
- **Data Science**: Correlation analysis, logistic regression, statistical testing
- **Data Engineering**: ETL pipelines, database design, API design
- **DevOps**: Docker, nginx, Makefile automation

**Built with Claude Code** - This project leverages [Claude Code](https://claude.com/claude-code) as an AI pair programmer.

## What It Does

- **Correlation Explorer**: See how event metrics (conflict count, Goldstein scores, media mentions) correlate with market returns across 33 symbols
- **Event Timeline**: Price charts overlaid with geopolitical event dots — red for conflict, green for cooperation
- **World Map**: Choropleth showing event intensity by country with drill-down details
- **Signals**: Two levels of market direction prediction:
  - **Level 1 (Historical Frequency)**: "When violent conflict events occur, oil went UP 72% of the time"
  - **Level 2 (Logistic Regression)**: "Based on today's event profile, probability of UP: 64%. Key drivers: Goldstein score, media coverage"

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     React Frontend (Vite)                        │
│  Dashboard │ Correlations │ Timeline │ World Map │ Signals       │
│  Recharts  │ React Query  │ Tailwind │ React Simple Maps         │
└──────────────────────────┬──────────────────────────────────────┘
                           │ /api proxy
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                                │
│  /api/events    │ /api/market     │ /api/analysis                │
│  /api/correlation │ /api/patterns │ /api/predictions             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌──────────────────────┐   ┌──────────────────────┐
│     PostgreSQL       │   │   Analysis Layer     │
│  Events │ MarketData │   │  Correlation         │
│  Links  │ Results    │   │  Historical Patterns │
└──────────────────────┘   │  Logistic Regression │
                           │  Event Study / CAR   │
                           │  Anomaly Detection   │
                           └──────────────────────┘

Data Sources: GDELT (events) + Yahoo Finance (prices)
```

## Quick Start

### Option 1: Docker (Recommended)

```bash
git clone https://github.com/YOUR_USERNAME/geopolitical-market-tracker.git
cd geopolitical-market-tracker

# Start all services (database, API, frontend)
make up

# Ingest data
make ingest-all

# Access:
#   Frontend:  http://localhost:3000
#   API Docs:  http://localhost:8000/docs
#   Database:  localhost:5432
```

### Option 2: Local Development

```bash
# Setup backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start database
make up-db
alembic upgrade head

# Setup frontend
cd frontend && npm install --legacy-peer-deps && cd ..

# Ingest data
make ingest-all

# Run (in separate terminals)
make dev-api       # Terminal 1: API on localhost:8000
make dev-frontend  # Terminal 2: React on localhost:3000
```

## API Endpoints

| Route | Method | Description |
|-------|--------|-------------|
| `/api/events` | GET | List events with filters (date, country, type) |
| `/api/events/map` | GET | Events aggregated by country for map |
| `/api/events/by-country` | GET | Event counts by country |
| `/api/events/by-type` | GET | Event counts by CAMEO type |
| `/api/market/{symbol}` | GET | Price data for a symbol |
| `/api/market/{symbol}/with-events` | GET | Price data merged with events |
| `/api/market/symbols` | GET | All tracked symbols |
| `/api/correlation/{symbol}` | GET | Correlations per event metric |
| `/api/correlation/{symbol}/rolling` | GET | Rolling correlation timeseries |
| `/api/correlation/top` | GET | Strongest event-market pairs |
| `/api/correlation/heatmap` | GET | Symbols x metrics matrix |
| `/api/patterns/{symbol}` | GET | Historical frequency pattern |
| `/api/patterns/{symbol}/all` | GET | All patterns for a symbol |
| `/api/predictions/logistic` | POST | Logistic regression prediction |
| `/api/predictions/logistic/{symbol}/summary` | GET | Model stats |
| `/api/analysis/regression/{symbol}` | GET | OLS regression results |
| `/api/analysis/event-study` | POST | Run event study |
| `/api/analysis/anomalies/detect` | GET | Anomaly detection |

Full interactive docs at `http://localhost:8000/docs`.

## Project Structure

```
geopolitical-market-tracker/
├── frontend/                      # React app (Vite + Tailwind)
│   └── src/
│       ├── api/                   # Axios client + React Query hooks
│       ├── components/
│       │   ├── layout/            # AppShell, Sidebar
│       │   ├── charts/            # PriceEventOverlay, Heatmap, Bar
│       │   ├── cards/             # MetricCard, PatternCard, PredictionCard
│       │   └── shared/            # SymbolSelector, DateRangePicker
│       ├── pages/                 # Dashboard, Correlation, Timeline, Map, Signals
│       └── lib/                   # Constants, formatters
│
├── src/                           # Python backend
│   ├── analysis/
│   │   ├── correlation.py         # Pearson/Spearman correlation
│   │   ├── historical_patterns.py # Level 1: conditional probability
│   │   ├── production_regression.py # OLS + Level 2: logistic regression
│   │   ├── production_event_study.py # Cumulative Abnormal Returns
│   │   ├── production_anomaly.py  # Z-score anomaly detection
│   │   └── feature_engineering.py # Shared data preparation
│   ├── api/
│   │   ├── main.py                # FastAPI app + middleware
│   │   ├── schemas.py             # Pydantic models
│   │   └── routes/                # 6 routers, 34 endpoints
│   ├── db/                        # SQLAlchemy models + queries
│   ├── config/                    # Constants, symbol mappings
│   └── ingestion/                 # GDELT + Yahoo Finance ETL
│
├── docker-compose.yml             # 4 services: db, api, frontend, migrate
├── Dockerfile.api                 # Python backend container
├── Dockerfile.frontend            # Multi-stage: Node build → nginx
├── nginx.conf                     # SPA routing + API proxy
├── Makefile                       # Dev and deployment commands
└── requirements.txt               # 16 Python packages
```

## Makefile Commands

```bash
# Docker
make up              # Start all services
make down            # Stop all services
make build           # Rebuild images
make logs            # View logs
make status          # Service status

# Development
make dev-api         # Run API with hot reload
make dev-frontend    # Run React dev server

# Data
make ingest-events   # Ingest GDELT events (7 days)
make ingest-market   # Ingest market data (30 days)
make ingest-all      # Both

# Testing
make test            # Run pytest

# Setup
make install         # Install Python deps
make install-frontend # Install frontend deps
make setup           # Full setup
```

## Tracked Symbols (33)

| Category | Symbols |
|----------|---------|
| Commodities | CL=F (Oil), BZ=F (Brent), GC=F (Gold), SI=F (Silver), NG=F (Gas), ZW=F (Wheat), ZC=F (Corn), ZS=F (Soybeans) |
| Currencies | EURUSD=X, USDJPY=X, GBPUSD=X, USDCNY=X, USDRUB=X, USDINR=X, USDBRL=X |
| ETFs | SPY, QQQ, EEM, VWO, EWZ, EWJ, FXI, EWG, EWT, EWY, INDA, XLE, XLF, GDX |
| Volatility | ^VIX |
| Bonds | TLT, IEF, HYG |

## Docker Services

| Service | Port | Description |
|---------|------|-------------|
| `db` | 5432 | PostgreSQL 15 |
| `api` | 8000 | FastAPI REST API |
| `frontend` | 3000 | React app via nginx |

## Data Sources

- **[GDELT](https://www.gdeltproject.org/)**: Global Database of Events, Language, and Tone
- **[Yahoo Finance](https://finance.yahoo.com/)**: Daily OHLCV data via yfinance

## License

MIT
