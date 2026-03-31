# Memory

Project state tracking for cross-device continuity.

## Current Status

**Phase G: MCP Server** - IN PROGRESS (next up)

## Architecture Pivot (2026-03-30)

Pivoting from custom agent/frontend to MCP + Telegram + Kalshi:
- Dropping: Anthropic API agent, LangGraph multi-agent, React frontend
- Adding: MCP server, Telegram bot, Kalshi integration
- Reason: Claude Max subscription eliminates need for API costs and custom chat UI

## Revised Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| G | MCP Server — expose 15+ tools for Claude Desktop/Code | Next |
| H | Cleanup — remove agent layer, frontend, unused deps | Planned |
| I | Kalshi — data source #6, trade tracking, 6 new MCP tools | Planned |
| J | Telegram Bot — daily briefings, anomaly alerts, watchlists | Planned |
| K | Testing & Documentation | Planned |

## Completed Phases (A–F)

- A: Framer Motion, API endpoints, Prediction Markets page
- B: Intelligence Briefing — 5-panel layout
- C: Framer Motion on all pages, FRED on Dashboard
- D: Sentiment analysis, pgvector, semantic search, glassmorphism UI
- E: RAG system, AI Summary panel, rag_search tool
- F: LangGraph multi-agent supervisor graph

## Data Sources (5 → 6)

1. GDELT - Geopolitical events
2. Yahoo Finance - Market prices (33 symbols)
3. RSS Feeds - News headlines
4. FRED - Economic indicators (6 series)
5. Polymarket - Prediction market odds
6. Kalshi - Event contracts + personal trades (planned)

## Final Target Stack

- Backend: FastAPI + SQLAlchemy + PostgreSQL + pgvector
- MCP Server: stdio transport, ~21 tools for Claude Desktop/Code
- Notifications: Telegram bot for mobile alerts
- Orchestration: Prefect
- No frontend, no Anthropic API key needed
