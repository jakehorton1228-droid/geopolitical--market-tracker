# Geopolitical Market Tracker

A full-stack intelligence platform that fuses geopolitical events, financial markets, economic indicators, news headlines, and prediction market odds into a unified analytical dashboard. It surfaces historical patterns, computes correlations, predicts market direction using interpretable models, and provides AI-powered analyst agents that can answer natural language questions about the data.

## About This Project

This is a personal project focused on building a complete full-stack system from scratch — React frontend, Python backend, PostgreSQL database, and an AI agent layer.

**Skills demonstrated:**
- **Frontend**: React 19, Vite, Tailwind CSS, Recharts, Framer Motion, React Query, react-simple-maps
- **Backend**: FastAPI, SQLAlchemy, Alembic, Pydantic
- **Data Science**: Correlation analysis, logistic regression, anomaly detection, event studies (CAR), statistical testing
- **Machine Learning**: Event-impact prediction — trained 5 models (Logistic Regression, Random Forest, XGBoost, LightGBM, MLP) on (significant event, affected asset) pairs with time-series aware train/val/test split, feature engineering, and automated champion promotion
- **MLOps**: MLflow experiment tracking and model registry, champion/challenger promotion, model loader with in-memory caching, weekly retraining via Prefect
- **Data Engineering**: ETL pipelines (5 data sources), event-centered feature engineering pipeline, database design, REST API design
- **AI Engineering**: Local LLM inference (Ollama + Gemma 4 26B MoE on Apple Silicon Metal GPU), LangGraph multi-agent orchestration, RAG pipelines, LangSmith observability, prompt engineering
- **UX**: Onboarding layer (inline tooltips + per-page help panels + plain-English glossary) so non-experts can navigate and interpret the data
- **DevOps**: Docker, Docker Compose, nginx, Prefect orchestration, Makefile automation

**Built with [Claude Code](https://claude.com/claude-code)** as an AI pair programmer.

## What It Does

### Pages

Pages are ordered in the sidebar to follow a natural user flow: start with today's briefing, explore where and when events happened, see what correlates, check predictions, then ask the AI freeform questions.

- **Intelligence Briefing** *(landing page)* — Flagship dashboard fusing all 5 data sources: FRED macro indicators, prediction market movers, fused event/price timeline, news headlines with sentiment, risk radar, and an AI-generated summary grounded in RAG retrieval
- **World Map** — Choropleth showing event intensity by country with drill-down details (Goldstein score, conflict/cooperation breakdown, media mentions)
- **Event Timeline** — Price charts overlaid with geopolitical event dots — red for conflict, green for cooperation. Drill-down table showing event details per day
- **Correlation Explorer** — Shows how event metrics (conflict count, Goldstein scores, media mentions) correlate with market returns across 33 symbols. Per-metric cards show strength (strong/moderate/weak) and significance in plain English, rolling correlation timeseries with confidence intervals, multi-symbol heatmap
- **Signals** — When a significant geopolitical event occurs, how do affected assets respond? Two levels:
  - **Level 1 (Historical Frequency)**: "When violent conflict events occur, oil went UP 72% of the time" — pure conditional probability counting
  - **Level 2 (Event-Impact ML Model)**: 5 classifiers trained on (significant event, affected asset) pairs — events filtered to |Goldstein| ≥ 5 AND ≥ 1000 mentions, target is 3-day forward return direction. Champion model served from MLflow with an **Active Model card** showing test AUC, model type, and deep-link to MLflow
- **Prediction Markets** — Browse geopolitical prediction markets from Polymarket — probabilities, 24h volume, sortable table with expandable probability trend charts
- **AI Analyst** — Chat interface powered by a LangGraph multi-agent pipeline with Gemma 4 26B MoE running locally via Ollama. Deterministic data collection and analysis, LLM-powered synthesis of intelligence assessments, fully traced in LangSmith
- **Dashboard** — Platform overview: event counts, tracked symbols, strongest correlations, FRED economic indicator strip with animated counters, and recent high-impact events table

Every page has a collapsible **"How to read this page"** panel explaining what the page is for, what to look for, and a glossary of domain terms used there. Every technical term (Goldstein scale, p-value, correlation coefficient, AUC, CAR, CAMEO, RAG, etc.) has an inline `?` tooltip with a plain-English definition pulled from a shared glossary.

### AI Analyst (LangGraph Pipeline)

A LangGraph intelligence pipeline that combines deterministic data operations with local LLM synthesis. The supervisor uses state-based routing (no LLM needed for routing), and only the final dissemination node calls the LLM — keeping the system fast, reliable, and cheap to run.

**Pipeline flow:**
```
Start -> Supervisor -> Collection -> Supervisor -> Analysis -> Supervisor -> Dissemination -> End
         (rules)      (determ.)                  (determ.)                  (LLM)
```

- **Supervisor** — Deterministic state-based routing (has data? has analysis? has response?)
- **Collection Node** — Query parsing + country/asset mappings fetch events, market data, headlines, and RAG context. No LLM tool selection.
- **Analysis Node** — Deterministic tool execution runs correlations, historical patterns, anomaly detection, and sentiment analysis on whatever was collected
- **Dissemination Node** — Gemma 4 26B MoE (via Ollama) synthesizes collected data and analysis into a structured intelligence briefing. This is the only LLM call in the pipeline.

**Observability:** Every graph invocation is traced in LangSmith — node executions, state transitions, and LLM generation are all logged and inspectable.

**Example questions:**
- "What are the top correlations for crude oil?"
- "What happened in Russia this month?"
- "Run a prediction for gold"
- "Detect anomalies for SPY over the last 90 days"

### ML Training Pipeline

The project trains and compares 5 machine learning models for event-impact prediction: given a significant geopolitical event, will the affected asset move UP over the next 3 trading days? The winning sklearn-compatible model is automatically registered in MLflow's Model Registry as the **champion** and served by the Signals page.

**Problem framing:**
- Filter to significant events: |Goldstein| ≥ 5 AND ≥ 1,000 media mentions
- Map each event to affected assets via `COUNTRY_ASSET_MAP` (e.g., Russia events → USDRUB=X, CL=F)
- Target: binary UP/DOWN based on cumulative return over the next 3 trading days
- Each training row represents one (event, asset) pair — not a calendar day

| Model | Type |
|-------|------|
| Logistic Regression | Linear baseline |
| Random Forest | Tree ensemble |
| XGBoost | Gradient boosting |
| LightGBM | Gradient boosting |
| MLP (PyTorch) | Neural network |

**Feature pipeline** (`analysis/event_features.py`):
- **Event features**: Goldstein score (magnitude + direction), media mentions, average tone, conflict/cooperation counts, event category flags (CAMEO-derived)
- **Asset context** (strictly lagged, no leakage): pre-event rolling volatility, momentum, volume
- **Sentiment context**: Recent FinBERT sentiment in the event's country

**Training** (`training/trainer.py`):
- Time-series aware train/val/test split (70/15/15, no leakage — chronological)
- Each model trained on the same event-centered dataset with the same StandardScaler
- All runs logged to MLflow: params, metrics (accuracy, precision, recall, F1, AUC-ROC on train/val/test splits), feature importance, classification reports, scaler + feature names as artifacts
- Apple Silicon MPS acceleration for PyTorch models
- **Champion promotion**: After training, the best sklearn-compatible model (highest test AUC) is registered in the MLflow Model Registry under the alias `champion`. Subsequent training runs only promote the new model if it beats the current champion's AUC (champion/challenger logic).

**Model loader** (`src/ml/model_loader.py`):
- Loads the champion from MLflow on first request, caches it in memory for subsequent requests
- Applies the exact same StandardScaler and feature ordering used during training
- Returns None gracefully if no champion is registered, so the app falls back to the legacy on-demand logistic regression

**Serving predictions** (`api/routes/predictions.py`):
- `POST /api/predictions/logistic` tries the champion first, falls back to the legacy `LogisticRegressionAnalyzer` if no champion exists
- `GET /api/predictions/logistic/{symbol}/summary` returns champion metadata (model type, AUC, version, feature list) when one exists

**Signals page integration:**
- An **Active Model card** in the top-right shows the serving model name, test AUC, champion badge, and a "View all models in MLflow →" link
- The Model Summary panel adapts its rendering depending on whether champion or legacy is serving (different fields in each response shape)

**Orchestration:** Weekly retraining via Prefect (`flows/training_flow.py`). View all runs at `http://localhost:5000` (MLflow UI).

**Latest training results (event-impact prediction, ~9.3K event-asset pairs):**

| Model | test AUC | Notes |
|-------|----------|-------|
| **mlp** | **0.540** | Best AUC |
| lightgbm | 0.504 | |
| xgboost | 0.502 | Champion (better calibrated) |
| random_forest | 0.492 | |
| logistic_regression | 0.458 | Baseline |

*These are honest numbers on a legitimately hard problem. Professional quants achieve 0.52–0.58 AUC on daily market direction prediction; anything higher usually indicates target leakage. The previous version of this pipeline reported 0.91 AUC due to rolling features that inadvertently included the target day's return — that has been fixed.*

### RAG System

The RAG pipeline provides contextual intelligence retrieval:
- **Embeddings**: 384-dimensional vectors (all-MiniLM-L6-v2) stored in pgvector
- **Retriever**: Cosine similarity search across headlines and events with distance thresholds
- **Context Builder**: Formats retrieved documents into structured LLM context
- **Briefing Context**: Multi-topic search across 5 geopolitical themes for situational awareness
- **AI Summary Panel**: Gemma 4 generated intelligence briefing on the Briefing page

### Onboarding Layer

The app assumes no prior knowledge of finance or geopolitics jargon. Every page is designed to be approachable to a first-time visitor:

- **Plain-English labels everywhere.** "Goldstein" becomes "Tone", "Mentions" becomes "Coverage", "Strongest Correlation (Pearson)" becomes "Strongest Event-Market Link."
- **Inline `?` tooltips** on every technical term. Hover to see a plain-English definition. Definitions are sourced from `frontend/src/utils/glossary.js` — one source of truth for 22 terms covering stats, ML, event metrics, and data sources.
- **Per-page "How to read this" panels** (collapsible). Each page explains what question it answers, what to look for, and defines the specific terms it uses. Collapsed by default so repeat users aren't slowed down.
- **Interpretation of raw numbers.** The Correlation Explorer doesn't just show `0.184, p=0.032, n=1234` — it shows "Moderate · significant" alongside the raw values so newcomers know how to read them.

### Sentiment Analysis

- FinBERT-based NLP scoring on news headlines (-1.0 to +1.0)
- Sentiment labels (positive/negative/neutral) with distribution tracking
- Daily sentiment trends and per-source breakdown
- Glassmorphism UI overhaul with sentiment-colored indicators

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
│  Briefing │ World Map │ Timeline │ Correlations │ Signals         │
│  Predictions │ Agent Chat │ Dashboard                              │
│  Recharts │ React Query │ Tailwind │ React Simple Maps            │
│  InfoTooltip + PageHelp onboarding layer                           │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ /api proxy (nginx)
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                               │
│  /api/events       │ /api/market        │ /api/analysis            │
│  /api/correlation  │ /api/patterns      │ /api/predictions*        │
│  /api/indicators   │ /api/headlines     │ /api/prediction-markets  │
│  /api/briefing     │ /api/agent/chat                                │
│                  * champion w/ legacy fallback                      │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           ▼                     ▼                      ▼
┌────────────────────┐  ┌────────────────┐  ┌────────────────────────┐
│  PostgreSQL +      │  │ Analysis Layer │  │   AI Agent Layer       │
│  pgvector          │  │ Correlation    │  │                        │
│                    │  │ Hist. Patterns │  │ LangGraph Pipeline:    │
│  Events            │  │ Logistic Reg.  │  │   Supervisor (rules)   │
│  MarketData        │  │ Event Study    │  │   Collection (determ.) │
│  NewsHeadlines     │  │ Anomaly Det.   │  │   Analysis (determ.)   │
│  EconomicIndicators│  │ Sentiment      │  │   Dissemination (LLM)  │
│  PredictionMarkets │  │ (FinBERT)      │  │                        │
│  CorrelationCache  │  │ ML Features    │  │ Gemma 4 26B MoE        │
│  AnalysisResults   │  │ (flat + seq)   │  │   via native Ollama    │
│  Embeddings (384d) │  │ RAG (pgvector) │  │   (Metal GPU accel.)   │
└────────────────────┘  └────────────────┘  │ LangSmith tracing      │
                                            └────────────────────────┘

Data Sources: GDELT + Yahoo Finance + RSS Feeds + FRED + Polymarket

┌─────────────────────────────────────────────────────────────────────┐
│              ML Training Pipeline & Model Registry                 │
│                                                                     │
│   Event Feature Pipeline (event-asset pairs, lagged features)      │
│        ↓                                                            │
│   Train 5 models: LogReg, RF, XGBoost, LightGBM, MLP               │
│        ↓                                                            │
│   Log runs to MLflow (params, metrics, artifacts, scaler)          │
│        ↓                                                            │
│   Champion promotion (best test AUC → Model Registry alias)        │
│        ↓                                                            │
│   Model Loader (src/ml) ← reads champion by alias                  │
│        ↓                                                            │
│   Signals page serves predictions from champion                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    Prefect Orchestration                           │
│  Daily Pipeline: Events → Market → RSS → FRED → Polymarket →      │
│                  Sentiment → Embeddings → Analysis                 │
│  Weekly Pipeline: Event Feature Engineering → 5-Model Training →   │
│                   MLflow → Champion Promotion                      │
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
#   FRED_API_KEY=...                (free at fred.stlouisfed.org)
#   LANGCHAIN_API_KEY=...           (optional, for LangSmith tracing)
#   OLLAMA_MODEL=gemma4:26b         (default — Gemma 4 26B MoE)

# Install Ollama natively for GPU acceleration (one-time)
brew install ollama

# Pull the LLM model (~18GB, first time only)
make pull-model

# Start Ollama (leave running in a dedicated terminal)
make ollama-serve

# Start all Docker services (database, API, frontend, MLflow, Prefect)
make start

# Ingest data from all 5 sources
make ingest-all

# Train ML models (optional — requires historical data)
make train

# Access:
#   Frontend:   http://localhost:3000
#   API Docs:   http://localhost:8000/docs
#   Prefect UI: http://localhost:4200
#   MLflow UI:  http://localhost:5000
#   Ollama:     http://localhost:11434
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
| **Briefing** | | |
| `/api/briefing/summary` | GET | AI-generated briefing via local LLM, or raw RAG context as fallback |
| **Agent** | | |
| `/api/agent/chat` | POST | Chat with LangGraph multi-agent intelligence pipeline |

Full interactive docs at `http://localhost:8000/docs`.

## Project Structure

```
geopolitical--market-tracker/
├── backend/
│   ├── src/
│   │   ├── agent/                      # AI agent module
│   │   │   ├── tools.py                # 15 tool definitions + execution dispatch
│   │   │   ├── synthesis.py            # Shared Ollama client for LLM-powered assessments
│   │   │   ├── graph.py                # LangGraph supervisor graph
│   │   │   ├── nodes.py                # Collection, Analysis, Dissemination agents
│   │   │   └── state.py                # Shared agent state schema
│   │   ├── rag/                        # RAG pipeline
│   │   │   ├── retriever.py            # pgvector similarity search
│   │   │   └── context.py              # Context formatting for LLM
│   │   ├── analysis/
│   │   │   ├── correlation.py          # Pearson/Spearman correlation
│   │   │   ├── historical_patterns.py  # Conditional probability analysis
│   │   │   ├── production_regression.py # OLS + logistic regression
│   │   │   ├── production_event_study.py # Cumulative Abnormal Returns
│   │   │   ├── production_anomaly.py   # Isolation Forest + Z-score
│   │   │   ├── sentiment.py            # FinBERT sentiment scoring
│   │   │   ├── embeddings.py           # Sentence transformer embeddings
│   │   │   ├── feature_engineering.py  # Shared data preparation
│   │   │   ├── event_features.py        # Event-centered ML feature pipeline
│   │   │   ├── ml_features.py          # Legacy ML feature pipeline (flat + windowed)
│   │   │   ├── synthesis.py            # Ollama client for LLM assessments
│   │   │   └── logging_config.py       # Logging utilities
│   │   ├── training/                   # ML model training
│   │   │   └── trainer.py              # 5-model comparison, MLflow logging,
│   │   │                               #   champion/challenger promotion
│   │   ├── ml/                         # ML runtime (inference layer)
│   │   │   └── model_loader.py         # Loads champion from MLflow registry,
│   │   │                               #   caches in memory, applies scaler
│   │   ├── api/
│   │   │   ├── main.py                 # FastAPI app + middleware
│   │   │   ├── schemas.py              # Pydantic request/response models
│   │   │   └── routes/                 # Routers (events, market, analysis,
│   │   │                               #   correlation, patterns, predictions,
│   │   │                               #   indicators, headlines, briefing,
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
│   │   ├── training_flow.py            # Weekly model training pipeline
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
│       │                               #   AnimatedNumber, Skeletons,
│       │                               #   InfoTooltip, PageHelp
│       ├── pages/                      # IntelligenceBriefing (landing),
│       │                               #   WorldMapView, EventTimeline,
│       │                               #   CorrelationExplorer, Signals,
│       │                               #   PredictionMarkets, AgentChat, Dashboard
│       └── utils/                      # Constants, formatters, animations,
│                                       #   glossary (22 plain-English term defs)
│
├── .env.example                        # Environment variable template
├── docs/                               # Architecture diagrams and roadmap
├── docker-compose.yml                  # Services: db, api, frontend, prefect, worker, ollama, mlflow
├── Dockerfile.frontend                 # Multi-stage: Node build → nginx
├── nginx.conf                          # SPA routing + API proxy
├── Makefile                            # Dev and deployment commands
├── SYMBOLS.md                          # Tracked symbols documentation
└── README.md
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
| `db` | 5432 | PostgreSQL 15 + pgvector |
| `api` | 8000 | FastAPI REST API |
| `frontend` | 3000 | React app via nginx |
| `prefect-server` | 4200 | Prefect UI + orchestration API |
| `prefect-worker` | — | Runs scheduled daily pipeline + weekly training |
| `mlflow` | 5000 | ML experiment tracking UI |

**Ollama runs natively on the host** (not in Docker) to use Metal GPU acceleration on Apple Silicon. Docker Desktop for Mac can't pass through the GPU, so containerized LLM inference is CPU-only. Running Ollama natively is 20-100x faster for models the size of Gemma 4 26B. Docker containers reach Ollama via `http://host.docker.internal:11434`.

## Development Phases

| Phase | Focus | Status |
|-------|-------|--------|
| A | Animation foundation, new API endpoints, Prediction Markets page | Done |
| B | Intelligence Briefing (6-panel layout, fused timeline, risk radar, AI summary) | Done |
| C | Visual polish — Framer Motion on all pages, FRED cards on Dashboard | Done |
| D | Sentiment analysis — pgvector, headline NLP, semantic search, glassmorphism UI | Done |
| E | RAG system — embeddings pipeline, vector search, context builder, AI Summary panel | Done |
| F | Multi-agent — LangGraph supervisor graph, specialist agents, chat UI | Done |
| G | Local LLM swap — replaced Anthropic API with Ollama + Gemma 4 26B MoE, deterministic collection/analysis nodes | Done |
| H | ML training pipeline — event-impact prediction, 5 models (LogReg, RF, XGBoost, LightGBM, MLP), MLflow tracking, target leakage audit | Done |
| I | LangSmith observability — full tracing on LangGraph pipeline | Done |
| J | Native Ollama (Metal GPU acceleration) — moved LLM out of Docker for 20-100x faster inference on Apple Silicon | Done |
| K | Onboarding layer — InfoTooltip + PageHelp components, glossary of 22 terms, plain-English labels, navigation reorder | Done |
| L | Champion model deployment — MLflow model registry with champion alias, model loader with caching, Signals page serves champion predictions with MLflow deep-link | Done |

## License

MIT
