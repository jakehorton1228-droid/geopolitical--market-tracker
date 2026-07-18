# Geopolitical Market Intelligence Platform

A data engineering–driven intelligence platform that fuses geopolitical events, financial markets, economic indicators, news sentiment, and prediction market odds through a **medallion architecture** (Bronze → Silver → Gold). **dbt models running on DuckDB** build the entire Silver + Gold medallion (DuckDB is the compute engine; Postgres stays the storage/serving layer), and domain-specific AI agents query the marts to deliver real-time intelligence briefings.

## About This Project

This is a portfolio project showcasing modern data engineering and AI agent systems. The pipeline is the product — it ingests from 5 live data sources, transforms through a three-layer medallion architecture, and serves analysis through both a REST API and an agentic LLM interface.

**Skills demonstrated:**
- **Data Engineering**: Medallion architecture (Bronze → Silver → Gold) built entirely with **dbt on DuckDB** — one engine, one lineage graph, right-sized (reads/writes Postgres via the postgres extension, no JVM). Seeds for reference data, incremental materialization, data quality testing, lineage tracking
- **Orchestration**: Prefect flows coordinating a 4-stage pipeline — ingest (extract-load) → enrich (ML) → transform (dbt) → analytics
- **SQL**: Window functions (LAG, rolling aggregates), CTEs, incremental models, dbt seeds, custom dbt tests
- **AI Engineering**: LangGraph multi-agent orchestration, local LLM inference (Ollama + Gemma 4 26B MoE), RAG with pgvector, LangSmith observability
- **Machine Learning**: Event-impact prediction (5 models), MLflow experiment tracking, champion/challenger promotion, time-series aware train/val/test splits
- **Backend**: FastAPI, SQLAlchemy, Alembic, Pydantic, PostgreSQL + pgvector
- **Frontend**: React 19, Vite, Tailwind CSS, Recharts, Framer Motion, React Query
- **DevOps**: Docker Compose, nginx, Makefile automation

**Built with [Claude Code](https://claude.com/claude-code)** as an AI pair programmer.

## Architecture: Medallion + Agents

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES (5)                           │
│  GDELT Events │ Yahoo Finance │ RSS Feeds │ FRED │ Polymarket      │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ Prefect daily ingestion
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    BRONZE (Raw Ingestion Tables)                   │
│  events │ market_data │ news_headlines │ economic_indicators │     │
│  prediction_markets                                                │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ dbt on DuckDB (+ seeds)
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 SILVER (Cleaned, Typed, Joined)                    │
│  silver_events          — Deduped, CAMEO classified, FIPS→ISO     │
│  silver_market          — Rolling returns (5d/20d), volatility,   │
│                           volume z-scores via SQL window functions │
│  silver_headlines       — Normalized sources, sentiment filtered   │
│  silver_event_market    — Cross-domain join via COUNTRY_ASSET_MAP  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ dbt models (incremental)
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GOLD (Business-Ready Marts)   [dbt on DuckDB]    │
│  gold_geopolitical_daily  — Daily event aggregates by country      │
│  gold_market_daily        — Market data + anomaly flags            │
│  gold_sentiment_daily     — Sentiment by source + "all" aggregate  │
│  gold_economic_snapshot   — FRED indicators with LAG() deltas      │
│  gold_daily_summary       — Cross-domain risk_level scoring        │
│                                                                     │
│  dbt tests: not_null, unique, accepted_values, custom assertions   │
│  dbt docs: full lineage DAG from source → gold                     │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           ▼                     ▼                      ▼
┌────────────────────┐  ┌────────────────┐  ┌────────────────────────┐
│  FastAPI REST API   │  │ Analysis Layer │  │   AI Agent Layer       │
│  49 endpoints       │  │ Correlation    │  │                        │
│  across 11 routers  │  │ Hist. Patterns │  │ LangGraph Pipeline:    │
│                     │  │ Anomaly Det.   │  │   Supervisor (rules)   │
│                     │  │ Event Study    │  │   Collection (determ.) │
│                     │  │ Sentiment      │  │   Analysis (determ.)   │
│                     │  │ (FinBERT)      │  │   Dissemination (LLM)  │
│                     │  │ RAG (pgvector) │  │                        │
│                     │  │                │  │ Gemma 4 26B MoE        │
│                     │  │                │  │   via native Ollama    │
│                     │  │                │  │ LangSmith tracing      │
└────────────────────┘  └────────────────┘  └────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    Prefect Orchestration                           │
│  Daily: Ingest → Enrich → Transform (dbt/DuckDB) → Analytics       │
│  Weekly: Event Feature Engineering → 5-Model Training → MLflow     │
└─────────────────────────────────────────────────────────────────────┘
```

## Medallion Architecture

### Bronze Layer (Raw Ingestion)

Five data sources land in raw tables via Prefect-orchestrated ingestion flows. Data arrives as-is from external APIs — deduplication at the DB level via unique constraints.

| Table | Source | Key |
|-------|--------|-----|
| `events` | GDELT | `global_event_id` |
| `market_data` | Yahoo Finance | `(symbol, date)` |
| `news_headlines` | RSS (Reuters, AP, BBC, Al Jazeera) | `url` |
| `economic_indicators` | FRED | `(series_id, date)` |
| `prediction_markets` | Polymarket | `(market_id, snapshot_date)` |

### Silver Layer (dbt models on DuckDB)

dbt models clean, type, enrich, and join Bronze data. dbt runs on **DuckDB**, which attaches Postgres via the postgres extension, reads Bronze, and materializes Silver back into Postgres (DuckDB is the compute engine; Postgres stays the storage layer). Reference maps (CAMEO groups, FIPS→ISO, country→asset) are **dbt seeds** generated from `constants.py`. Right-sized for this ~1.5M-row dataset: no JVM, no cluster. The SQL ports to MotherDuck/Snowflake/BigQuery if the data ever outgrows a single node.

| Model | What it does | Key SQL features |
|-----------|-------------|------------------|
| `silver_events` | Deterministic dedup on `global_event_id`, classify via CAMEO → `event_group`, normalize FIPS → ISO country codes, add `is_significant` flag | `row_number()` dedup, map lookups via `LEFT JOIN`, `COALESCE` |
| `silver_market` | Dedup on `(symbol, date)`, compute rolling returns (5d, 20d), rolling volatility, volume z-score | window frames (`ROWS BETWEEN`), `LAG()`, `STDDEV_SAMP()` |
| `silver_headlines` | Dedup on URL, normalize source names, filter to sentiment-scored headlines | `row_number()`, lookup `LEFT JOIN`, `COALESCE` |
| `silver_event_market` | Join events → market via `COUNTRY_ASSET_MAP`, aggregate to one row per (date, country, symbol) | multi-table join, deterministic dominant-group, conditional aggregation |

### Gold Layer (dbt Models)

dbt builds business-ready marts from the Silver models (same dbt-on-DuckDB project) with incremental materialization, schema tests, and documentation. Run `dbt docs serve` to view the full lineage DAG — one graph spanning seeds → Bronze sources → Silver → Gold.

| Model | Grain | Key Metrics | Tested |
|-------|-------|-------------|--------|
| `gold_geopolitical_daily` | (date, country) | conflict/cooperation/violent event counts, Goldstein stats, mention totals | `not_null`, `accepted_values` on event_group |
| `gold_market_daily` | (date, symbol) | close, returns, volatility, anomaly flags | `not_null` on close |
| `gold_sentiment_daily` | (date, source) | headline count, avg sentiment, most extreme headlines | `accepted_values` on source |
| `gold_economic_snapshot` | (date, series_id) | value, previous_value, change, change_pct (via `LAG()`) | `accepted_values` on series_id |
| `gold_daily_summary` | (date) | cross-domain risk_level (low/elevated/high) via custom macro | `unique`, `accepted_values` on risk_level |

**Data quality**: Schema tests (`not_null`, `unique`, `accepted_values`, `relationships`) plus 3 custom tests (`assert_no_null_dates`, `assert_positive_mention_counts`, `assert_valid_risk_levels`).

## What It Does

### Pages

- **Intelligence Briefing** *(landing)* — Fuses all 5 sources: FRED macro strip, prediction market movers, event/price timeline, sentiment-colored headlines, risk radar, AI-generated summary
- **World Map** — Choropleth of event intensity by country with Goldstein breakdown
- **Event Timeline** — Price charts overlaid with geopolitical event dots (conflict red, cooperation green)
- **Correlation Explorer** — Event metric ↔ market return correlations with rolling timeseries and heatmap
- **Signals** — Two-level prediction: historical frequency patterns + ML model (champion from MLflow)
- **Prediction Markets** — Polymarket odds, volume, probability trends
- **AI Analyst** — Chat with LangGraph agent pipeline (Gemma 4 26B via Ollama)
- **Dashboard** — Platform overview with event counts, top correlations, FRED indicators

Every page has collapsible help panels and inline `?` tooltips on domain terms.

### AI Analyst (LangGraph Pipeline)

```
Start → Supervisor → Collection → Supervisor → Analysis → Supervisor → Dissemination → End
         (rules)      (determ.)                 (determ.)                (LLM)
```

- **Supervisor** — Deterministic state-based routing (no LLM)
- **Collection** — Fetches events, market data, headlines, RAG context via tool dispatch
- **Analysis** — Runs correlations, patterns, anomaly detection deterministically
- **Dissemination** — Gemma 4 26B synthesizes a structured intelligence briefing (only LLM call)

Fully traced in LangSmith.

### ML Training Pipeline

5 models trained on event-impact prediction: given a significant event (|Goldstein| ≥ 5, ≥ 1000 mentions), will the affected asset move UP over 3 days?

| Model | Test AUC |
|-------|----------|
| MLP (PyTorch) | 0.540 |
| LightGBM | 0.504 |
| XGBoost (champion) | 0.502 |
| Random Forest | 0.492 |
| Logistic Regression | 0.458 |

Time-series aware split (70/15/15), MLflow tracking, champion/challenger promotion, weekly retraining via Prefect.

## Data Sources (5)

| Source | What It Captures | Update Frequency | Orientation |
|--------|-----------------|------------------|-------------|
| [GDELT](https://www.gdeltproject.org/) | Geopolitical events (conflicts, diplomacy, protests) | Daily | Backward-looking |
| [Yahoo Finance](https://finance.yahoo.com/) | Market prices for 33 symbols (OHLCV) | Daily | Current |
| RSS Feeds | News headlines (Reuters, AP, BBC, Al Jazeera) | Every run | Backward-looking |
| [FRED](https://fred.stlouisfed.org/) | Economic indicators (GDP, CPI, unemployment, rates) | Daily/Monthly/Quarterly | Lagging |
| [Polymarket](https://polymarket.com/) | Prediction market odds for geopolitical events | Daily snapshots | Forward-looking |

## Quick Start

### Docker (Recommended)

```bash
git clone https://github.com/jakehorton1228-droid/geopolitical--market-tracker.git
cd geopolitical--market-tracker

# Copy .env.example and configure
cp .env.example .env
# Set: FRED_API_KEY, LANGCHAIN_API_KEY (optional), OLLAMA_MODEL=gemma4:26b

# Install Ollama natively for GPU acceleration
brew install ollama
make pull-model       # ~18GB, first time only
make ollama-serve     # Leave running in a dedicated terminal

# Start all Docker services
make start

# Ingest data + run transforms
make ingest-all       # Fetch from all 5 sources
make transforms       # dbt build: Silver + Gold on DuckDB

# Optional
make train            # Train ML models (requires historical data)
make dbt-docs         # View dbt lineage DAG

# Access:
#   Frontend:   http://localhost:3000
#   API Docs:   http://localhost:8000/docs
#   Prefect UI: http://localhost:4200
#   MLflow UI:  http://localhost:5000
```

### Local Development

```bash
# libomp is XGBoost's OpenMP runtime (needed to import/train the champion model)
brew install libomp

python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt

make up-db
cd backend && alembic upgrade head && cd ..
cd frontend && npm install --legacy-peer-deps && cd ..

make ingest-all
make dev-api          # Terminal 1
make dev-frontend     # Terminal 2
```

## Project Structure

```
geopolitical--market-tracker/
├── backend/
│   ├── src/
│   │   ├── agent/                         # LangGraph AI agent pipeline
│   │   ├── analysis/                      # Correlation, patterns, anomaly, sentiment, RAG
│   │   ├── training/                      # ML model training + MLflow
│   │   ├── ml/                            # Model loader (champion from MLflow)
│   │   ├── api/                           # FastAPI app + 11 routers
│   │   ├── rag/                           # pgvector retrieval + context
│   │   ├── db/                            # SQLAlchemy models + queries
│   │   ├── config/                        # Settings, constants, symbol mappings
│   │   └── ingestion/                     # GDELT, Yahoo Finance, RSS, FRED, Polymarket (pure extract-load)
│   ├── dbt_project/                       # dbt-on-DuckDB medallion (Silver + Gold)
│   │   ├── models/
│   │   │   ├── silver/                    # 4 Silver models (dedup, classify, join, window fns)
│   │   │   ├── staging/                   # Ephemeral passthroughs (lineage)
│   │   │   └── gold/                      # 5 Gold mart models (incremental)
│   │   ├── seeds/                         # CAMEO/FIPS/country-asset reference CSVs
│   │   ├── tests/                         # Custom data quality tests
│   │   ├── macros/                        # risk_level scoring + generate_schema_name
│   │   ├── profiles.yml                   # dbt-duckdb: attaches Postgres as `pg`
│   │   └── dbt_project.yml
│   ├── flows/                             # Prefect orchestration (4 stages)
│   │   ├── ingestion_flow.py              # 1. Ingest — extract-load, 5 sources
│   │   ├── enrich_flow.py                 # 2. Enrich — sentiment (FinBERT) + embeddings
│   │   ├── transform_flow.py              # 3. Transform — dbt build (Silver + Gold)
│   │   ├── analysis_flow.py               # 4. Analytics — correlations + patterns
│   │   ├── daily_pipeline.py              # Master: ingest → enrich → transform → analytics
│   │   └── training_flow.py               # Weekly ML training
│   ├── notebooks/                         # DuckDB data-exploration notebook
│   ├── alembic/                           # Database migrations
│   └── requirements.txt
├── frontend/                              # React 19 + Vite + Tailwind
│   └── src/
│       ├── pages/                         # 8 pages (Briefing, Map, Timeline, etc.)
│       ├── components/                    # Charts, cards, shared UI
│       ├── api/                           # Axios + React Query hooks
│       └── utils/                         # Glossary, formatters, animations
├── docker-compose.yml                     # db, api, frontend, prefect, mlflow
├── Makefile                               # 30+ automation targets
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

## Docker Services

| Service | Port | Description |
|---------|------|-------------|
| `db` | 5432 | PostgreSQL 15 + pgvector |
| `api` | 8000 | FastAPI REST API |
| `frontend` | 3000 | React app via nginx |
| `prefect-server` | 4200 | Prefect UI + orchestration API |
| `prefect-worker` | — | Runs scheduled pipelines |
| `mlflow` | 5000 | ML experiment tracking UI |

**Ollama runs natively on the host** (not in Docker) for Metal GPU acceleration on Apple Silicon.

## Makefile Commands

```bash
make start              # Start all Docker services
make ingest-all         # Ingest from all 5 sources (extract-load)
make transforms         # dbt build: Silver + Gold on DuckDB
make pipeline           # Full daily pipeline (ingest → enrich → transform → analytics)
make train              # Train ML models, log to MLflow
make dbt-run            # Run dbt models (Silver + Gold)
make dbt-test           # Run dbt data quality tests
make dbt-docs           # Generate + serve dbt lineage DAG
make dev-api            # Run API locally (dev mode)
make dev-frontend       # Run React frontend (dev mode)
make test               # Run pytest suite
```

## License

MIT
