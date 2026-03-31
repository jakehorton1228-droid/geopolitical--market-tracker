# Geopolitical Market Tracker

A personal intelligence platform that fuses geopolitical events, financial markets, economic indicators, news sentiment, and prediction market data into a unified analytical system. It ingests data from 6 sources, runs statistical analysis (correlations, regressions, anomaly detection), and exposes everything through an MCP server — so Claude can query it all via natural language.

Daily briefings and anomaly alerts are pushed to Telegram.

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                     Data Sources (6)                            │
│  GDELT  │  Yahoo Finance  │  RSS Feeds  │  FRED  │  Polymarket │
│                           │  Kalshi                             │
└──────────────────────┬──────────────────────────────────────────┘
                       │  Prefect daily pipeline
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PostgreSQL + pgvector                          │
│  Events │ MarketData │ Headlines │ Indicators │ Predictions    │
│  Kalshi Positions │ Fills │ Settlements │ Embeddings (384d)    │
│  CorrelationCache │ AnalysisResults                            │
└──────────┬────────────────────────────┬─────────────────────────┘
           │                            │
           ▼                            ▼
┌─────────────────────┐     ┌──────────────────────────┐
│    FastAPI           │     │     Analysis Layer       │
│  REST API endpoints  │     │  Correlation (Pearson/   │
│  /api/events         │     │    Spearman)             │
│  /api/market         │     │  Historical patterns     │
│  /api/correlation    │     │  Logistic regression     │
│  /api/predictions    │     │  Anomaly detection       │
│  /api/headlines      │     │    (Isolation Forest +   │
│  /api/indicators     │     │     Z-score)             │
│  /api/kalshi         │     │  Event studies (CAR)     │
│  /api/briefing       │     │  Sentiment (FinBERT)     │
│  ...                 │     │  RAG (pgvector search)   │
└─────────────────────┘     └──────────────────────────┘
           │                            │
           └────────────┬───────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MCP Server (stdio)                           │
│  ~21 tools exposed to Claude Desktop / Claude Code             │
│  Events, market data, correlations, predictions, anomalies,    │
│  sentiment, RAG search, Kalshi positions/P&L, watchlists       │
└──────────────────────┬──────────────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼                         ▼
┌──────────────────┐     ┌────────────────────┐
│  Claude Desktop  │     │   Telegram Bot     │
│  Claude Code     │     │  Daily briefings   │
│  Natural language│     │  Anomaly alerts    │
│  queries         │     │  P&L updates       │
└──────────────────┘     └────────────────────┘
```

**No frontend, no API keys for AI.** Claude Desktop/Code connects to the MCP server directly using the Claude Max subscription. Telegram handles mobile notifications.

## Skills Demonstrated

- **Backend**: FastAPI, SQLAlchemy, Alembic, Pydantic, PostgreSQL
- **Data Engineering**: ETL pipelines across 6 heterogeneous sources, pgvector embeddings, database design
- **Data Science**: Correlation analysis, logistic regression, anomaly detection (Isolation Forest + Z-score), event studies (CAR), statistical testing
- **AI/ML Engineering**: MCP server development, RAG pipeline (embeddings + vector similarity search + context builder), NLP sentiment analysis (FinBERT)
- **DevOps**: Docker, Docker Compose, Prefect orchestration, Makefile automation
- **Integration**: Telegram Bot API, Kalshi trading API, GDELT, Yahoo Finance, FRED, Polymarket, RSS

**Built with [Claude Code](https://claude.com/claude-code)** as an AI pair programmer.

## Data Sources

| Source | What It Captures | Update Frequency |
|--------|-----------------|------------------|
| [GDELT](https://www.gdeltproject.org/) | Geopolitical events — conflicts, diplomacy, protests | Daily |
| [Yahoo Finance](https://finance.yahoo.com/) | Market prices for 33 instruments (OHLCV) | Daily |
| RSS Feeds | News headlines from Reuters, AP, BBC, Al Jazeera | Multiple times daily |
| [FRED](https://fred.stlouisfed.org/) | Economic indicators — GDP, CPI, unemployment, Fed rate, 10Y yield, consumer sentiment | Daily/Monthly/Quarterly |
| [Polymarket](https://polymarket.com/) | Prediction market odds for geopolitical events | Daily snapshots |
| [Kalshi](https://kalshi.com/) | Event contract positions, trades, settlements, P&L | Daily + on-demand |

## MCP Tools

The MCP server exposes ~21 tools that Claude can call via natural language:

| Tool | What It Does |
|------|-------------|
| `get_recent_events` | Query GDELT events by country, date range, event type |
| `get_event_summary` | Event counts grouped by country or type |
| `get_market_data` | OHLCV + returns for any of 33 tracked instruments |
| `get_correlations` | Event-market correlation analysis for a symbol |
| `get_top_correlations` | Strongest correlation pairs across all symbols |
| `get_historical_patterns` | "When X happens, Y goes UP Z% of the time" |
| `run_prediction` | Logistic regression market direction prediction |
| `detect_anomalies` | Isolation Forest + Z-score anomaly detection |
| `list_symbols` | All 33 tracked instruments by category |
| `get_symbol_countries` | Country-to-symbol sensitivity mappings |
| `get_headline_sentiment` | News headlines with FinBERT sentiment scores |
| `get_sentiment_summary` | Aggregate sentiment statistics and trends |
| `search_similar_content` | Semantic vector search over headlines and events |
| `rag_search` | RAG retrieval formatted as an intelligence briefing |
| `get_briefing_context` | Multi-topic situational awareness context |
| `get_kalshi_positions` | Current Kalshi positions with unrealized P&L |
| `get_kalshi_pnl` | Aggregate and per-position profit/loss |
| `get_kalshi_trades` | Trade fill history with filters |
| `get_kalshi_markets` | Browse active Kalshi event contracts |
| `get_kalshi_market_detail` | Detailed view of a single market |
| `cross_reference_predictions` | Compare Kalshi vs Polymarket probabilities |

**Example queries in Claude Desktop:**
- "What are the top correlations for crude oil?"
- "What happened in Russia this week?"
- "Run a prediction for gold based on recent events"
- "Detect anomalies for SPY over the last 90 days"
- "What's my current Kalshi P&L?"
- "Are there any Kalshi markets related to the conflict spike in the Middle East?"

## Analysis Capabilities

- **Correlation Analysis**: Pearson and Spearman correlations between geopolitical event metrics (conflict count, Goldstein scores, media mentions) and market returns across 33 symbols. Rolling correlation timeseries with confidence intervals.
- **Historical Patterns**: Conditional probability analysis — "When violent conflict events occur, oil went UP 72% of the time" with sample sizes and statistical significance.
- **Market Prediction**: Logistic regression for next-day market direction using event features (Goldstein score, media coverage, event type). Reports probability, key drivers, and model diagnostics.
- **Anomaly Detection**: Isolation Forest + Z-score hybrid approach to identify unusual market-event mismatches.
- **Event Studies**: Cumulative Abnormal Return (CAR) analysis around geopolitical events.
- **Sentiment Analysis**: FinBERT-based NLP scoring on news headlines with sentiment trends and aggregation.
- **RAG Search**: Vector similarity search (pgvector, 384-dim embeddings) across headlines and events for contextual intelligence retrieval.

## Tracked Instruments (33)

| Category | Symbols |
|----------|---------|
| Commodities | CL=F (Oil), BZ=F (Brent), GC=F (Gold), SI=F (Silver), NG=F (Gas), ZW=F (Wheat), ZC=F (Corn), ZS=F (Soybeans) |
| Currencies | EURUSD=X, USDJPY=X, GBPUSD=X, USDCNY=X, USDRUB=X, USDINR=X, USDBRL=X |
| ETFs | SPY, QQQ, EEM, VWO, EWZ, EWJ, FXI, EWG, EWT, EWY, INDA, XLE, XLF, GDX |
| Volatility | ^VIX |
| Bonds | TLT, IEF, HYG |

## Quick Start

```bash
git clone <repo-url>
cd geopolitical--market-tracker

# Configure environment
cp .env.example .env
# Edit .env:
#   FRED_API_KEY=...           (free at fred.stlouisfed.org)
#   KALSHI_API_KEY=...         (from kalshi.com)
#   KALSHI_EMAIL=...
#   TELEGRAM_BOT_TOKEN=...    (from @BotFather)
#   TELEGRAM_CHAT_ID=...

# Start services
make start    # PostgreSQL + FastAPI + Prefect
make migrate  # Run database migrations
make ingest-all  # Ingest from all 6 sources

# Connect MCP server to Claude Desktop
# Add to ~/Library/Application Support/Claude/claude_desktop_config.json:
# {
#   "mcpServers": {
#     "geomarket": {
#       "command": "python",
#       "args": ["-m", "src.mcp_server"],
#       "cwd": "/path/to/backend",
#       "env": { "DATABASE_URL": "postgresql://user:pass@localhost:5432/geomarket" }
#     }
#   }
# }

# Or use with Claude Code (auto-configured via .mcp.json)
```

## Project Structure

```
geopolitical--market-tracker/
├── backend/
│   ├── src/
│   │   ├── mcp_server/              # MCP server for Claude Desktop/Code
│   │   │   ├── server.py            # Tool registration + stdio transport
│   │   │   └── __main__.py          # Entry point
│   │   ├── agent/
│   │   │   └── tools.py             # ~21 tool implementations
│   │   ├── rag/                     # RAG pipeline
│   │   │   ├── retriever.py         # pgvector similarity search
│   │   │   └── context.py           # Context formatting for LLM
│   │   ├── analysis/                # Statistical analysis modules
│   │   │   ├── correlation.py       # Pearson/Spearman correlation
│   │   │   ├── historical_patterns.py
│   │   │   ├── production_regression.py
│   │   │   ├── production_anomaly.py
│   │   │   ├── production_event_study.py
│   │   │   └── embeddings.py        # FinBERT + sentence transformers
│   │   ├── telegram/                # Telegram bot for notifications
│   │   │   └── bot.py               # Briefings, alerts, P&L updates
│   │   ├── api/                     # FastAPI REST API
│   │   │   ├── main.py
│   │   │   ├── schemas.py
│   │   │   └── routes/
│   │   ├── db/                      # SQLAlchemy models + queries
│   │   ├── config/                  # Settings, constants, symbol mappings
│   │   └── ingestion/               # ETL for all 6 data sources
│   │       ├── gdelt.py
│   │       ├── market_data.py
│   │       ├── rss_feeds.py
│   │       ├── fred.py
│   │       ├── polymarket.py
│   │       └── kalshi.py
│   ├── flows/                       # Prefect orchestration
│   │   ├── daily_pipeline.py        # Master: ingest → analyze → notify
│   │   ├── ingestion_flow.py
│   │   └── analysis_flow.py
│   ├── tests/
│   ├── alembic/                     # Database migrations
│   └── requirements.txt
│
├── .mcp.json                        # Claude Code MCP config
├── docker-compose.yml
├── Makefile
└── README.md
```

## Docker Services

| Service | Port | Description |
|---------|------|-------------|
| `db` | 5432 | PostgreSQL 15 + pgvector |
| `api` | 8000 | FastAPI REST API |
| `prefect-server` | 4200 | Prefect orchestration UI |
| `prefect-worker` | — | Runs daily pipeline |

## Development

| Phase | Focus | Status |
|-------|-------|--------|
| A | Data ingestion, API endpoints, initial analysis | Done |
| B | Intelligence briefing system | Done |
| C | Animation polish, economic indicators | Done |
| D | Sentiment analysis, pgvector embeddings, semantic search | Done |
| E | RAG pipeline, vector search, context builder | Done |
| F | Multi-agent orchestration (LangGraph) | Done |
| G | MCP server — Claude Desktop/Code integration | Done |
| H | Architecture cleanup — removed frontend + agent layer | Done |
| I | Kalshi integration — positions, P&L, trade analysis | Done |
| J | Telegram bot — daily briefings, anomaly alerts | Done |
| K | Testing and documentation | Done |

## License

MIT
