# Geopolitical Market Tracker

A full-stack application that correlates global geopolitical events with financial market movements. It surfaces historical patterns, computes correlations, predicts market direction using interpretable models, and provides an AI-powered analyst agent that can answer natural language questions about the data.

## About This Project

This is a personal project focused on building a complete full-stack system from scratch — React frontend, Python backend, PostgreSQL database, and an AI agent layer.

**Skills practiced:**
- **Frontend**: React, Vite, Tailwind CSS, Recharts, React Query
- **Backend**: FastAPI, SQLAlchemy, Pydantic
- **Data Science**: Correlation analysis, logistic regression, anomaly detection, statistical testing
- **Data Engineering**: ETL pipelines, database design, API design
- **AI Engineering**: Claude API tool use, agentic loops, prompt engineering
- **DevOps**: Docker, nginx, Prefect orchestration, Makefile automation

**Built with Claude Code** — This project leverages [Claude Code](https://claude.com/claude-code) as an AI pair programmer.

## What It Does

- **Dashboard**: Overview of event counts, tracked symbols, strongest correlations, and recent high-impact events
- **Correlation Explorer**: See how event metrics (conflict count, Goldstein scores, media mentions) correlate with market returns across 33 symbols
- **Event Timeline**: Price charts overlaid with geopolitical event dots — red for conflict, green for cooperation
- **World Map**: Choropleth showing event intensity by country with drill-down details
- **Signals**: Two levels of market direction prediction:
  - **Level 1 (Historical Frequency)**: "When violent conflict events occur, oil went UP 72% of the time"
  - **Level 2 (Logistic Regression)**: "Based on today's event profile, probability of UP: 64%. Key drivers: Goldstein score, media coverage"
- **AI Agent**: Chat interface powered by Claude that can query events, run correlations, analyze patterns, make predictions, and detect anomalies using natural language

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       React Frontend (Vite)                         │
│  Dashboard │ Correlations │ Timeline │ World Map │ Signals │ Agent  │
│  Recharts  │ React Query  │ Tailwind │ React Simple Maps           │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ /api proxy (nginx)
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                                │
│  /api/events      │ /api/market       │ /api/analysis              │
│  /api/correlation  │ /api/patterns    │ /api/predictions           │
│  /api/agent/chat  (Claude tool use)                                │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                ┌────────────────┼────────────────┐
                ▼                ▼                 ▼
┌────────────────────┐  ┌────────────────┐  ┌────────────────────┐
│    PostgreSQL      │  │ Analysis Layer │  │    AI Agent        │
│  Events            │  │ Correlation    │  │ Claude API         │
│  MarketData        │  │ Hist. Patterns │  │ 10 tools           │
│  CorrelationCache  │  │ Logistic Reg.  │  │ Calls analysis     │
│  AnalysisResults   │  │ Event Study    │  │ functions directly  │
└────────────────────┘  │ Anomaly Det.   │  └────────────────────┘
                        └────────────────┘

Data Sources: GDELT (events) + Yahoo Finance (prices)

┌─────────────────────────────────────────────────────────────────────┐
│                    Prefect Orchestration                             │
│  Daily Pipeline: Ingest Events → Ingest Market → Run Analysis       │
│  Prefect UI: http://localhost:4200                                  │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Option 1: Docker (Recommended)

```bash
git clone https://github.com/YOUR_USERNAME/geopolitical-market-tracker.git
cd geopolitical-market-tracker

# (Optional) Enable AI agent — copy .env.example and add your key
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...

# Start all services (database, API, frontend)
make up

# Ingest data
make ingest-all

# Access:
#   Frontend:   http://localhost:3000
#   API Docs:   http://localhost:8000/docs
#   Prefect UI: http://localhost:4200
#   Database:   localhost:5432
```

### Option 2: Local Development

```bash
# Setup backend
python -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt

# Start database
make up-db
cd backend && alembic upgrade head && cd ..

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
| **Events** | | |
| `/api/events` | GET | List events with filters (date, country, type, min_mentions) |
| `/api/events/count` | GET | Total event count for a date range |
| `/api/events/map` | GET | Events aggregated by country for map visualization |
| `/api/events/by-country` | GET | Event counts grouped by country |
| `/api/events/by-type` | GET | Event counts grouped by CAMEO event type |
| `/api/events/{event_id}` | GET | Single event by ID |
| **Market** | | |
| `/api/market` | GET | Market data for a symbol with date range |
| `/api/market/{symbol}` | GET | Price data (OHLCV + returns) for a symbol |
| `/api/market/{symbol}/with-events` | GET | Price data merged with geopolitical events |
| `/api/market/{symbol}/latest` | GET | Most recent price for a symbol |
| `/api/market/{symbol}/returns` | GET | Return series for a symbol |
| `/api/market/{symbol}/stats` | GET | Summary statistics for a symbol |
| `/api/market/symbols` | GET | All tracked symbols with metadata |
| `/api/market/symbols/flat` | GET | Flat list of ticker symbols |
| **Correlation** | | |
| `/api/correlation/{symbol}` | GET | Correlations per event metric for a symbol |
| `/api/correlation/{symbol}/rolling` | GET | Rolling correlation timeseries |
| `/api/correlation/top` | GET | Strongest event-market correlation pairs |
| `/api/correlation/heatmap` | GET | Symbols x event metrics correlation matrix |
| **Patterns** | | |
| `/api/patterns/{symbol}` | GET | Historical frequency pattern for a symbol |
| `/api/patterns/{symbol}/all` | GET | All event group patterns for a symbol |
| **Predictions** | | |
| `/api/predictions/logistic` | POST | Logistic regression market direction prediction |
| `/api/predictions/logistic/{symbol}/summary` | GET | Model fit statistics |
| **Analysis** | | |
| `/api/analysis/results` | GET | Analysis results with filters |
| `/api/analysis/significant` | GET | Statistically significant results |
| `/api/analysis/anomalies` | GET | Detected anomalies |
| `/api/analysis/summary` | GET | Overall analysis summary |
| `/api/analysis/regression/{symbol}` | GET | OLS regression results |
| `/api/analysis/event-study` | POST | Run cumulative abnormal return study |
| `/api/analysis/anomalies/detect` | GET | Run anomaly detection for a symbol |
| **Agent** | | |
| `/api/agent/chat` | POST | Chat with the AI analyst (requires ANTHROPIC_API_KEY) |

Full interactive docs at `http://localhost:8000/docs`.

## AI Agent

The AI Agent is a Claude-powered analyst that can answer natural language questions about the data. It uses Claude's tool use capability to call 10 internal analysis functions:

| Tool | Description |
|------|-------------|
| `get_recent_events` | Query GDELT events by country, date, type |
| `get_event_summary` | Event counts by country or type |
| `get_market_data` | OHLCV + returns for any symbol |
| `get_correlations` | Event-market correlation for a symbol |
| `get_top_correlations` | Strongest correlations across all symbols |
| `get_historical_patterns` | Frequency patterns ("when X, Y goes UP Z%") |
| `run_prediction` | Logistic regression market direction prediction |
| `detect_anomalies` | Isolation Forest + Z-score anomaly detection |
| `list_symbols` | All 33 tracked instruments |
| `get_symbol_countries` | Country-symbol sensitivity mappings |

**Setup**: Set `ANTHROPIC_API_KEY` in `.env`. Without it, the rest of the app works normally — the agent endpoint returns a helpful 503 message.

**Example questions**:
- "What are the top correlations for crude oil?"
- "What happened in Russia this month?"
- "Run a prediction for gold"
- "Detect anomalies for SPY over the last 90 days"

## Project Structure

```
geopolitical-market-tracker/
├── backend/                            # Python backend
│   ├── src/
│   │   ├── agent/                      # AI agent module
│   │   │   ├── tools.py                # 10 tool definitions + execution dispatch
│   │   │   └── service.py              # Claude API agentic loop
│   │   ├── analysis/
│   │   │   ├── correlation.py          # Pearson/Spearman correlation
│   │   │   ├── historical_patterns.py  # Level 1: conditional probability
│   │   │   ├── production_regression.py # OLS + Level 2: logistic regression
│   │   │   ├── production_event_study.py # Cumulative Abnormal Returns
│   │   │   ├── production_anomaly.py   # Isolation Forest + Z-score anomalies
│   │   │   ├── feature_engineering.py  # Shared data preparation
│   │   │   └── logging_config.py       # Logging utilities
│   │   ├── api/
│   │   │   ├── main.py                 # FastAPI app + middleware
│   │   │   ├── schemas.py              # Pydantic request/response models
│   │   │   └── routes/                 # 7 routers (events, market, analysis,
│   │   │                               #   correlation, patterns, predictions, agent)
│   │   ├── db/                         # SQLAlchemy models + queries
│   │   ├── config/                     # Settings, constants, symbol mappings
│   │   └── ingestion/                  # GDELT + Yahoo Finance ETL
│   ├── tests/                          # pytest test suites
│   ├── alembic/                        # Database migrations
│   ├── scripts/                        # CLI utilities
│   ├── flows/                          # Prefect pipeline flows
│   │   ├── ingestion_flow.py           # Daily GDELT + Yahoo Finance ingestion
│   │   ├── analysis_flow.py            # Daily correlation + pattern computation
│   │   ├── daily_pipeline.py           # Master flow (ingestion → analysis)
│   │   └── deploy.py                   # Registers cron schedule
│   ├── Dockerfile.api                  # API container
│   ├── Dockerfile.worker               # Prefect worker container
│   ├── alembic.ini
│   └── requirements.txt
│
├── frontend/                           # React app (Vite + Tailwind)
│   └── src/
│       ├── api/                        # Axios client + React Query hooks
│       │                               #   (events, market, correlation, patterns,
│       │                               #    predictions, agent)
│       ├── components/
│       │   ├── layout/                 # AppShell, Sidebar
│       │   ├── charts/                 # PriceEventOverlay, Heatmap, Bar
│       │   ├── cards/                  # MetricCard, PatternCard, PredictionCard
│       │   └── shared/                 # SymbolSelector, DateRangePicker, Skeletons
│       ├── pages/                      # Dashboard, Correlation, Timeline,
│       │                               #   Map, Signals, AgentChat
│       └── lib/                        # Constants, formatters
│
├── .env.example                        # Environment variable template
├── docker-compose.yml                  # 6 services: db, api, frontend, prefect, worker, migrate
├── Dockerfile.frontend                 # Multi-stage: Node build → nginx
├── nginx.conf                          # SPA routing + API proxy
├── Makefile                            # Dev and deployment commands
├── SYMBOLS.md                          # Tracked symbols documentation
└── README.md
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

# Pipeline
make pipeline        # Run daily pipeline manually
make prefect-logs    # View Prefect worker logs

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
| `prefect-server` | 4200 | Prefect UI + orchestration API |
| `prefect-worker` | — | Runs scheduled daily pipeline |

## Data Sources

- **[GDELT](https://www.gdeltproject.org/)**: Global Database of Events, Language, and Tone
- **[Yahoo Finance](https://finance.yahoo.com/)**: Daily OHLCV data via yfinance

## License

MIT
