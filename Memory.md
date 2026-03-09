# Memory

Project state tracking for cross-device continuity.

## Current Status

**Phase B: Intelligence Briefing** - DONE

## Completed Phases

### Phase B (Done) - 2026-03-08
- Created `IntelligenceBriefing.jsx` with 5-panel layout
- Added `/briefing` route and sidebar navigation
- All panels use existing API endpoints (no backend changes needed)
- 5 panels implemented:
  1. FRED Macro Strip - Economic indicators with AnimatedNumber counters
  2. Prediction Market Movers - 24h probability changes from Polymarket
  3. Fused Event/Price Timeline - CL=F price chart with event dots overlay
  4. News Headlines - RSS headlines from Reuters, AP, BBC, Al Jazeera
  5. Risk Radar - Top 8 countries by event count (30d) with heat coloring

### Phase A (Done)
- Framer Motion animations foundation
- New API endpoints for indicators, headlines, prediction markets
- Prediction Markets page with sortable table and probability charts

### Phase C (Done)
- Framer Motion animations on all pages
- FRED economic indicators strip on Dashboard with animated counters

## Next Up

### Phase D: Sentiment Analysis
- pgvector extension for embeddings
- NLP sentiment scoring on news headlines
- Sentiment colors on Briefing headlines panel

### Phase E: RAG System
- Embeddings pipeline for events and headlines
- Vector similarity search
- Context builder for AI agent

## Data Sources (5)

1. GDELT - Geopolitical events
2. Yahoo Finance - Market prices (33 symbols)
3. RSS Feeds - News headlines
4. FRED - Economic indicators (6 series)
5. Polymarket - Prediction market odds

## Architecture Notes

- Backend: FastAPI + SQLAlchemy + PostgreSQL
- Frontend: React 19 + Vite + Tailwind + Recharts + Framer Motion
- Orchestration: Prefect
- AI Agent: Claude API with 10 tools
