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
	@echo "  Ollama:     http://localhost:11434"
	@echo "  Database:   localhost:5432"
	@echo ""
	@echo "$(YELLOW)If this is your first run, pull the LLM model:$(NC)"
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

pull-model: ## Pull Llama 3 model into Ollama
	@echo "$(BLUE)Pulling Llama 3.1 8B model (this may take a few minutes)...$(NC)"
	docker exec gmt-ollama ollama pull llama3.1:8b
	@echo "$(GREEN)Model ready.$(NC)"

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

ingest-all: ingest-events ingest-market ingest-headlines ingest-fred ingest-polymarket ## Ingest all 5 data sources

ingest-history: ## Ingest 1 year of historical data (events + market)
	@echo "$(BLUE)Ingesting 1 year of historical data...$(NC)"
	@echo "$(YELLOW)This may take 10-20 minutes...$(NC)"
	docker exec gmt-api python -c "\
from datetime import date, timedelta; \
from src.ingestion.gdelt import GDELTIngestion; \
from src.ingestion.market_data import MarketDataIngestion; \
print('Ingesting GDELT events (1 year)...'); \
g = GDELTIngestion(); \
g.ingest_date_range(date.today() - timedelta(days=365), date.today() - timedelta(days=1)); \
print('Ingesting market data (1 year)...'); \
m = MarketDataIngestion(); \
m.ingest_all_symbols(date.today() - timedelta(days=365), date.today(), skip_existing=False); \
"
	@echo "$(GREEN)Historical data ingestion complete.$(NC)"

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

logs-ollama: ## View Ollama logs
	docker compose logs -f ollama

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
