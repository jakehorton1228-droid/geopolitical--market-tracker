# Geopolitical Market Tracker

A full-stack intelligence platform that fuses geopolitical events, financial markets, economic indicators, news headlines, and prediction market odds into a unified analytical dashboard. It surfaces historical patterns, computes correlations, predicts market direction using interpretable models, and provides an AI-powered analyst agent that can answer natural language questions about the data.

## About This Project

This is a personal project focused on building a complete full-stack system from scratch — React frontend, Python backend, PostgreSQL database, and an AI agent layer.

**Skills practiced:**
- **Frontend**: React 19, Vite, Tailwind CSS, Recharts, Framer Motion, React Query, react-simple-maps
- **Backend**: FastAPI, SQLAlchemy, Alembic, Pydantic
- **Data Science**: Correlation analysis, logistic regression, anomaly detection, statistical testing
- **Data Engineering**: ETL pipelines (5 data sources), database design, REST API design
- **AI Engineering**: Claude API tool use, agentic loops, prompt engineering
- **DevOps**: Docker, Docker Compose, nginx, Prefect orchestration, Makefile automation

**Built with Claude Code** — This project leverages [Claude Code](https://claude.com/claude-code) as an AI pair programmer.

## What It Does

### Pages

- **Intelligence Briefing** *(planned)*: Flagship dashboard fusing all 5 data sources — FRED macro indicators, prediction market movers, fused event/price timeline, news headlines with sentiment, and a risk radar showing which countries/topics are heating up
- **Dashboard**: Overview of event counts, tracked symbols, strongest correlations, FRED economic indicator strip with animated counters, and recent high-impact events table
- **Prediction Markets**: Browse geopolitical prediction markets from Polymarket — probabilities, 24h volume, sortable table with expandable probability trend charts
- **Correlation Explorer**: See how event metrics (conflict count, Goldstein scores, media mentions) correlate with market returns across 33 symbols. Includes rolling correlation timeseries with confidence intervals and a multi-symbol heatmap
- **Event Timeline**: Price charts overlaid with geopolitical event dots — red for conflict, green for cooperation. Drill-down table showing event details per day
- **World Map**: Choropleth showing event intensity by country with drill-down details (Goldstein score, conflict/cooperation breakdown, media mentions)
- **Signals**: Two levels of market direction prediction:
  - **Level 1 (Historical Frequency)**: "When violent conflict events occur, oil went UP 72% of the time"
  - **Level 2 (Logistic Regression)**: "Based on today's event profile, probability of UP: 64%. Key drivers: Goldstein score, media coverage"
- **AI Agent**: Chat interface powered by Claude that can query events, run correlations, analyze patterns, make predictions, and detect anomalies using natural language

### Planned Capabilities

- **Sentiment Analysis**: NLP sentiment scoring on news headlines using pgvector embeddings
- **RAG System**: Vector similarity search over events and headlines for contextual AI responses
- **Multi-Agent System**: LangGraph-orchestrated specialist agents (Collection, Analysis, Dissemination) with a supervisor graph that routes between them
- **Automated Workflows**: Prefect-triggered daily briefings, anomaly alerts with conditional routing, and a user-defined watchlist system
- **Real-Time Updates**: WebSocket streaming for live data, LangGraph streaming for agent chat, and a live-updating briefing page
- **Human-in-the-Loop**: LangGraph `interrupt()` review gates on generated briefings and alerts before dissemination
- **MCP Server**: Standardized Model Context Protocol interface exposing all intelligence tools for use by external AI systems

### AI Agent

The AI Agent is a Claude-powered analyst that uses tool use to call 10 internal analysis functions:

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

### Animations

All pages use Framer Motion for polished UI transitions:
- Page-level transitions via AnimatePresence on route changes
- Staggered entrance animations on cards, table rows, and list items
- Spring-animated number counters on the FRED indicator strip
- Expand/collapse panels with smooth height animation
- Hover/tap micro-interactions on interactive elements
- Animated probability bars on the Prediction Markets page

## Data Sources (5)

| Source | What It Captures | Update Frequency | Time Horizon |
|--------|-----------------|------------------|--------------|
| [GDELT](https://www.gdeltproject.org/) | Geopolitical events (conflicts, diplomacy, protests) | Daily | Backward-looking |
| [Yahoo Finance](https://finance.yahoo.com/) | Market prices for 33 symbols (OHLCV) | Daily | Current |
| RSS Feeds | News headlines (Reuters, AP, BBC, Al Jazeera) | Every run | Backward-looking |
| [FRED](https://fred.stlouisfed.org/) | Economic indicators (GDP, CPI, unemployment, Fed rate, 10Y yield, consumer sentiment) | Daily/Monthly/Quarterly | Lagging |
| [Polymarket](https://polymarket.com/) | Prediction market odds for geopolitical events | Daily snapshots | Forward-looking |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    React Frontend (Vite + Framer Motion)           │
│  Dashboard │ Predictions │ Correlations │ Timeline │ Map           │
│  Signals │ Agent Chat                                             │
│  Recharts │ React Query │ Tailwind │ React Simple Maps            │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ /api proxy (nginx)
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                               │
│  /api/events       │ /api/market        │ /api/analysis            │
│  /api/correlation  │ /api/patterns      │ /api/predictions         │
│  /api/indicators   │ /api/headlines     │ /api/prediction-markets  │
│  /api/briefing     │ /api/risk          │ /api/agent/chat          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                ┌────────────────┼────────────────┐
                ▼                ▼                 ▼
┌────────────────────┐  ┌────────────────┐  ┌────────────────────┐
│    PostgreSQL      │  │ Analysis Layer │  │    AI Agent        │
│  Events            │  │ Correlation    │  │ Claude API         │
│  MarketData        │  │ Hist. Patterns │  │ 10 tools           │
│  NewsHeadlines     │  │ Logistic Reg.  │  │ Calls analysis     │
│  EconomicIndicators│  │ Event Study    │  │ functions directly  │
│  PredictionMarkets │  │ Anomaly Det.   │  │                    │
│  CorrelationCache  │  └────────────────┘  └────────────────────┘
│  AnalysisResults   │
└────────────────────┘

Data Sources: GDELT + Yahoo Finance + RSS Feeds + FRED + Polymarket

┌─────────────────────────────────────────────────────────────────────┐
│                    Prefect Orchestration                           │
│  Daily Pipeline: Events → Market → RSS → FRED → Polymarket →     │
│                  Analysis                                         │
│  Prefect UI: http://localhost:4200                                │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Option 1: Docker (Recommended)

```bash
git clone https://github.com/jakehorton1228-droid/geopolitical--market-tracker.git
cd geopolitical--market-tracker

# Copy .env.example and add your keys
cp .env.example .env
# Edit .env and set:
#   ANTHROPIC_API_KEY=sk-ant-...  (for AI agent)
#   FRED_API_KEY=...              (for economic indicators, free at fred.stlouisfed.org)

# Start all services (database, API, frontend)
make start

# Run migrations
make migrate

# Ingest data from all 5 sources
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
| **Indicators** | | |
| `/api/indicators/latest` | GET | Latest value + delta for each FRED series |
| `/api/indicators/{series_id}` | GET | Time series for a single indicator |
| **Headlines** | | |
| `/api/headlines/recent` | GET | Recent news headlines, filterable by source |
| **Prediction Markets** | | |
| `/api/prediction-markets` | GET | All tracked Polymarket markets with latest odds |
| `/api/prediction-markets/movers` | GET | Markets with biggest probability changes |
| `/api/prediction-markets/{market_id}/history` | GET | Probability time series |
| **Analysis** | | |
| `/api/analysis/results` | GET | Analysis results with filters |
| `/api/analysis/significant` | GET | Statistically significant results |
| `/api/analysis/anomalies` | GET | Detected anomalies |
| `/api/analysis/summary` | GET | Overall analysis summary |
| `/api/analysis/regression/{symbol}` | GET | OLS regression results |
| `/api/analysis/event-study` | POST | Run cumulative abnormal return study |
| `/api/analysis/anomalies/detect` | GET | Run anomaly detection for a symbol |
| **Briefing** *(planned)* | | |
| `/api/briefing/snapshot` | GET | Composite briefing data (indicators + movers + headlines + risk) |
| **Risk** *(planned)* | | |
| `/api/risk/heatmap` | GET | Country-level risk scoring with temporal trends |
| **Agent** | | |
| `/api/agent/chat` | POST | Chat with the AI analyst (requires ANTHROPIC_API_KEY) |

Full interactive docs at `http://localhost:8000/docs`.

## Project Structure

```
geopolitical--market-tracker/
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
│   │   │   └── routes/                 # Routers (events, market, analysis,
│   │   │                               #   correlation, patterns, predictions,
│   │   │                               #   indicators, headlines,
│   │   │                               #   prediction_markets, agent)
│   │   ├── db/                         # SQLAlchemy models + queries
│   │   ├── config/                     # Settings, constants, symbol mappings
│   │   └── ingestion/                  # GDELT, Yahoo Finance, RSS, FRED, Polymarket
│   ├── tests/                          # pytest test suites
│   ├── alembic/                        # Database migrations
│   ├── scripts/                        # CLI utilities
│   ├── flows/                          # Prefect pipeline flows
│   │   ├── ingestion_flow.py           # Daily ingestion (5 sources)
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
│       ├── components/
│       │   ├── layout/                 # AppShell (AnimatePresence), Sidebar
│       │   ├── charts/                 # PriceEventOverlay, CorrelationHeatmap,
│       │   │                           #   TopCorrelationsBar
│       │   ├── cards/                  # MetricCard, PatternCard, PredictionCard
│       │   └── shared/                 # SymbolSelector, DateRangePicker,
│       │                               #   AnimatedNumber, Skeletons
│       ├── pages/                      # Dashboard, PredictionMarkets,
│       │                               #   CorrelationExplorer, EventTimeline,
│       │                               #   WorldMapView, Signals, AgentChat
│       └── utils/                      # Constants, formatters, animation presets
│
├── .env.example                        # Environment variable template
├── docker-compose.yml                  # Services: db, api, frontend, prefect, worker, migrate
├── Dockerfile.frontend                 # Multi-stage: Node build → nginx
├── nginx.conf                          # SPA routing + API proxy
├── Makefile                            # Dev and deployment commands
├── SYMBOLS.md                          # Tracked symbols documentation
└── README.md
```

## Makefile Commands

```bash
# Docker
make start           # Start all services (alias: make up)
make stop            # Stop all services (alias: make down)
make build           # Rebuild images
make logs            # View logs
make status          # Service status
make migrate         # Run database migrations

# Development
make dev-api         # Run API with hot reload
make dev-frontend    # Run React dev server

# Data
make ingest-events   # Ingest GDELT events (7 days)
make ingest-market   # Ingest market data (30 days)
make ingest-all      # All sources

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

## FRED Indicators (6)

| Series | FRED ID | Frequency |
|--------|---------|-----------|
| Gross Domestic Product | GDP | Quarterly |
| Consumer Price Index | CPIAUCSL | Monthly |
| Unemployment Rate | UNRATE | Monthly |
| Federal Funds Rate | DFF | Daily |
| 10-Year Treasury Yield | DGS10 | Daily |
| Consumer Sentiment (Michigan) | UMCSENT | Monthly |

## Docker Services

| Service | Port | Description |
|---------|------|-------------|
| `db` | 5432 | PostgreSQL 15 |
| `api` | 8000 | FastAPI REST API |
| `frontend` | 3000 | React app via nginx |
| `prefect-server` | 4200 | Prefect UI + orchestration API |
| `prefect-worker` | — | Runs scheduled daily pipeline |

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| A | Animation foundation, new API endpoints, Prediction Markets page | Done |
| B | Intelligence Briefing v1 (5-panel layout, fused timeline, risk radar) | Planned |
| C | Visual polish — Framer Motion on all pages, FRED cards on Dashboard | Done |
| D | Sentiment analysis — pgvector, headline NLP, sentiment colors | Planned |
| E | RAG system — embeddings pipeline, vector search, context builder | Planned |
| F | Multi-agent — LangGraph, Collection/Analysis/Dissemination agents | Planned |
| G | Automated workflows — daily briefings, anomaly alerts, watchlists | Planned |
| H | Real-time — WebSockets, streaming agent chat, live briefing | Planned |
| I | MCP server — standardized tool interface for external AI | Planned |
| J | Testing and documentation | Planned |

## License

MIT
