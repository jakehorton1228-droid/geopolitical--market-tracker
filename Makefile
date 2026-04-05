# Geopolitical Market Tracker - Makefile
#
# Usage:
#   make help       - Show available commands
#   make up         - Start all services
#   make down       - Stop all services
#   make dev        - Run API + frontend in dev mode
#
# ============================================================================

.PHONY: help up down start stop restart logs build clean migrate \
        dev-api dev-frontend dev test lint \
        ingest-events ingest-market ingest-all \
        pipeline prefect-logs train pull-model

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Backend directory
BACKEND := backend

# Activate venv with backend on PYTHONPATH
ACTIVATE := . venv/bin/activate && PYTHONPATH=$(BACKEND)

# ============================================================================
# HELP
# ============================================================================

help: ## Show this help message
	@echo ""
	@echo "$(BLUE)Geopolitical Market Tracker$(NC)"
	@echo "=============================="
	@echo ""
	@echo "$(GREEN)Docker Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(up|down|start|stop|restart|logs|build|clean|migrate|status)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Development Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(dev)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Data Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(ingest)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Other Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -vE '(up|down|start|stop|restart|logs|build|clean|migrate|status|dev|ingest)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

# ============================================================================
# DOCKER - FULL STACK
# ============================================================================

up: ## Start all services (database, API, frontend)
	@echo "$(GREEN)Starting all services...$(NC)"
	docker compose up -d
	@echo "$(YELLOW)Waiting for database to be healthy...$(NC)"
	@sleep 5
	@$(MAKE) --no-print-directory _init-db
	@echo ""
	@echo "$(GREEN)Services ready!$(NC)"
	@echo "  Frontend:   http://localhost:3000"
	@echo "  API Docs:   http://localhost:8000/docs"
	@echo "  Prefect UI: http://localhost:4200"
	@echo "  MLflow UI:  http://localhost:5000"
	@echo "  Database:   localhost:5432"
	@echo ""
	@echo "$(YELLOW)Don't forget to start native Ollama in another terminal:$(NC)"
	@echo "  make ollama-serve"
	@echo ""
	@echo "$(YELLOW)First run? Pull the model:$(NC)"
	@echo "  make pull-model"

_init-db: ## (internal) Run migrations if needed
	@echo "$(BLUE)Running database migrations...$(NC)"
	@docker exec gmt-api alembic upgrade head 2>/dev/null || echo "$(YELLOW)Migrations already applied$(NC)"
	@EVENT_COUNT=$$(docker exec gmt-db psql -U postgres -d geopolitical_tracker -t -c "SELECT COUNT(*) FROM events;" 2>/dev/null | tr -d ' ' || echo "0"); \
	if [ "$$EVENT_COUNT" = "0" ] || [ -z "$$EVENT_COUNT" ]; then \
		echo "$(YELLOW)No events found. Run 'make ingest-all' to populate data.$(NC)"; \
	else \
		echo "$(GREEN)Database already has $$EVENT_COUNT events.$(NC)"; \
	fi

down: ## Stop all services
	@echo "$(YELLOW)Stopping all services...$(NC)"
	docker compose down
	@echo "$(GREEN)All services stopped.$(NC)"

start: up ## Alias for 'up'

stop: down ## Alias for 'down'

restart: ## Restart all services
	@echo "$(YELLOW)Restarting all services...$(NC)"
	docker compose restart
	@echo "$(GREEN)All services restarted.$(NC)"

logs: ## View logs from all services (follow mode)
	docker compose logs -f

logs-api: ## View API logs only
	docker compose logs -f api

logs-db: ## View database logs only
	docker compose logs -f db

build: ## Rebuild all Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	docker compose build
	@echo "$(GREEN)Build complete.$(NC)"

clean: ## Stop services and remove volumes (WARNING: deletes data!)
	@echo "$(RED)WARNING: This will delete all data!$(NC)"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	docker compose down -v
	@echo "$(GREEN)Cleaned up.$(NC)"

migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	docker compose --profile setup up migrate
	@echo "$(GREEN)Migrations complete.$(NC)"

status: ## Show status of all services
	docker compose ps

# ============================================================================
# DOCKER - INDIVIDUAL SERVICES
# ============================================================================

up-db: ## Start only the database
	docker compose up -d db
	@echo "$(GREEN)Database started on localhost:5432$(NC)"

up-api: ## Start database and API
	docker compose up -d db api
	@echo "$(GREEN)API started on http://localhost:8000$(NC)"

pull-model: ## Pull Gemma 4 model into native Ollama (requires ollama installed on host)
	@echo "$(BLUE)Pulling Gemma 4 26B MoE model (~18GB, this may take several minutes)...$(NC)"
	@command -v ollama >/dev/null 2>&1 || { echo "$(RED)Ollama not installed. Run: brew install ollama$(NC)"; exit 1; }
	ollama pull gemma4:26b
	@echo "$(GREEN)Model ready.$(NC)"

ollama-serve: ## Start native Ollama server (run this in a separate terminal)
	@echo "$(BLUE)Starting native Ollama server (uses Metal GPU on Apple Silicon)...$(NC)"
	@command -v ollama >/dev/null 2>&1 || { echo "$(RED)Ollama not installed. Run: brew install ollama$(NC)"; exit 1; }
	ollama serve

# ============================================================================
# LOCAL DEVELOPMENT (without Docker for API/frontend, DB via Docker)
# ============================================================================

dev-api: ## Run API locally (requires venv and running DB)
	@echo "$(BLUE)Starting API in development mode...$(NC)"
	$(ACTIVATE) uvicorn src.api.main:app --reload --port 8000

dev-frontend: ## Run React frontend in dev mode
	@echo "$(BLUE)Starting frontend in development mode...$(NC)"
	cd frontend && npm run dev

dev: ## Instructions for running both API and frontend locally
	@echo "$(YELLOW)Run these commands in separate terminals:$(NC)"
	@echo ""
	@echo "  Terminal 1 (API):      make dev-api"
	@echo "  Terminal 2 (Frontend): make dev-frontend"
	@echo ""

# ============================================================================
# DATA INGESTION
# ============================================================================

ingest-events: ## Ingest GDELT events (last 7 days)
	@echo "$(BLUE)Ingesting GDELT events...$(NC)"
	docker exec gmt-api python -c "\
from datetime import date, timedelta; \
from src.ingestion.gdelt import GDELTIngestion; \
g = GDELTIngestion(); \
g.ingest_date_range(date.today() - timedelta(days=7), date.today() - timedelta(days=1)); \
"
	@echo "$(GREEN)Event ingestion complete.$(NC)"

ingest-market: ## Ingest market data (last 30 days)
	@echo "$(BLUE)Ingesting market data...$(NC)"
	docker exec gmt-api python -c "\
from datetime import date, timedelta; \
from src.ingestion.market_data import MarketDataIngestion; \
m = MarketDataIngestion(); \
m.ingest_all_symbols(date.today() - timedelta(days=30), date.today()); \
"
	@echo "$(GREEN)Market data ingestion complete.$(NC)"

ingest-headlines: ## Ingest RSS headlines (last 3 days)
	@echo "$(BLUE)Ingesting RSS headlines...$(NC)"
	docker exec gmt-api python -c "\
from src.ingestion.rss_feeds import RSSIngestion; \
r = RSSIngestion(); \
r.ingest_all_feeds(); \
"
	@echo "$(GREEN)Headlines ingestion complete.$(NC)"

ingest-fred: ## Ingest FRED economic indicators
	@echo "$(BLUE)Ingesting FRED indicators...$(NC)"
	docker exec gmt-api python -c "\
from src.ingestion.fred import FREDIngestion; \
f = FREDIngestion(); \
f.ingest_all_series(); \
"
	@echo "$(GREEN)FRED ingestion complete.$(NC)"

ingest-polymarket: ## Ingest Polymarket prediction markets
	@echo "$(BLUE)Ingesting Polymarket data...$(NC)"
	docker exec gmt-api python -c "\
from src.ingestion.polymarket import PolymarketIngestion; \
p = PolymarketIngestion(); \
p.ingest_markets(); \
"
	@echo "$(GREEN)Polymarket ingestion complete.$(NC)"

post-ingest: ## Run sentiment scoring and embedding generation on new data
	@echo "$(BLUE)Scoring sentiment and generating embeddings...$(NC)"
	docker exec gmt-api python -c "\
from src.db.connection import get_session; \
from src.analysis.sentiment import score_unprocessed_headlines; \
from src.analysis.embeddings import embed_unprocessed_headlines, embed_unprocessed_events; \
session = get_session().__enter__(); \
s = score_unprocessed_headlines(session); \
h = embed_unprocessed_headlines(session); \
e = embed_unprocessed_events(session); \
print(f'Scored {s} headlines, embedded {h} headlines, embedded {e} events'); \
"
	@echo "$(GREEN)NLP processing complete.$(NC)"

ingest-all: ingest-events ingest-market ingest-headlines ingest-fred ingest-polymarket post-ingest ## Ingest all 5 data sources + NLP processing

ingest-history: ## Ingest 5 years of historical data (events + market + FRED)
	@echo "$(BLUE)Ingesting 5 years of historical data...$(NC)"
	@echo "$(YELLOW)GDELT alone will take 1-3 hours. Grab a coffee.$(NC)"
	docker exec gmt-api python -c "\
from datetime import date, timedelta; \
from src.ingestion.gdelt import GDELTIngestion; \
from src.ingestion.market_data import MarketDataIngestion; \
from src.ingestion.fred import FREDIngestion; \
print('Ingesting GDELT events (5 years)...'); \
g = GDELTIngestion(); \
g.ingest_date_range(date.today() - timedelta(days=1825), date.today() - timedelta(days=1)); \
print('Ingesting market data (5 years)...'); \
m = MarketDataIngestion(); \
m.ingest_all_symbols(date.today() - timedelta(days=1825), date.today(), skip_existing=False); \
print('Ingesting FRED indicators (5 years)...'); \
f = FREDIngestion(); \
f.ingest_all_series(start_date=date.today() - timedelta(days=1825), end_date=date.today()); \
"
	@echo "$(GREEN)Historical data ingestion complete. Run 'make post-ingest' to score sentiment and generate embeddings.$(NC)"

ingest-history-1y: ## Ingest 1 year of historical data (faster, for testing)
	@echo "$(BLUE)Ingesting 1 year of historical data...$(NC)"
	@echo "$(YELLOW)This may take 15-30 minutes...$(NC)"
	docker exec gmt-api python -c "\
from datetime import date, timedelta; \
from src.ingestion.gdelt import GDELTIngestion; \
from src.ingestion.market_data import MarketDataIngestion; \
from src.ingestion.fred import FREDIngestion; \
print('Ingesting GDELT events (1 year)...'); \
g = GDELTIngestion(); \
g.ingest_date_range(date.today() - timedelta(days=365), date.today() - timedelta(days=1)); \
print('Ingesting market data (1 year)...'); \
m = MarketDataIngestion(); \
m.ingest_all_symbols(date.today() - timedelta(days=365), date.today(), skip_existing=False); \
print('Ingesting FRED indicators (1 year)...'); \
f = FREDIngestion(); \
f.ingest_all_series(start_date=date.today() - timedelta(days=365), end_date=date.today()); \
"
	@echo "$(GREEN)Historical data ingestion complete. Run 'make post-ingest' to score sentiment and generate embeddings.$(NC)"

# ============================================================================
# PREFECT PIPELINE
# ============================================================================

pipeline: ## Run the daily pipeline manually (ingestion + analysis)
	@echo "$(BLUE)Running daily pipeline...$(NC)"
	docker exec gmt-api python -m flows.daily_pipeline
	@echo "$(GREEN)Pipeline complete.$(NC)"

train: ## Train all ML models and log to MLflow
	@echo "$(BLUE)Running model training pipeline...$(NC)"
	docker exec gmt-api python -m flows.training_flow
	@echo "$(GREEN)Training complete. View results at http://localhost:5000$(NC)"

prefect-logs: ## View Prefect worker logs
	docker compose logs -f prefect-worker

logs-mlflow: ## View MLflow logs
	docker compose logs -f mlflow

# ============================================================================
# TESTING & LINTING
# ============================================================================

test: ## Run tests
	@echo "$(BLUE)Running tests...$(NC)"
	$(ACTIVATE) pytest $(BACKEND)/tests/ -v

lint: ## Run linters
	@echo "$(BLUE)Running linters...$(NC)"
	$(ACTIVATE) black --check $(BACKEND)/src/
	$(ACTIVATE) flake8 $(BACKEND)/src/

format: ## Format code with black
	@echo "$(BLUE)Formatting code...$(NC)"
	$(ACTIVATE) black $(BACKEND)/src/
	@echo "$(GREEN)Formatting complete.$(NC)"

# ============================================================================
# SETUP
# ============================================================================

install: ## Install Python dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	pip install -r $(BACKEND)/requirements.txt
	@echo "$(GREEN)Dependencies installed.$(NC)"

install-frontend: ## Install frontend dependencies
	@echo "$(BLUE)Installing frontend dependencies...$(NC)"
	cd frontend && npm install --legacy-peer-deps
	@echo "$(GREEN)Frontend dependencies installed.$(NC)"

setup: ## Full setup: install deps, start DB, run migrations
	@echo "$(BLUE)Running full setup...$(NC)"
	$(MAKE) install
	$(MAKE) up-db
	@echo "$(YELLOW)Waiting for database to be ready...$(NC)"
	sleep 5
	. venv/bin/activate && cd $(BACKEND) && alembic upgrade head
	@echo "$(GREEN)Setup complete! Run 'make up' to start all services.$(NC)"
