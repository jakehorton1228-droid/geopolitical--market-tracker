# Geopolitical Market Tracker - Makefile
#
# Usage:
#   make help       - Show available commands
#   make up         - Start all services
#   make down       - Stop all services
#   make logs       - View logs
#
# ============================================================================

.PHONY: help up down start stop restart logs build clean migrate \
        dev-api dev-dashboard dev test lint \
        ingest-events ingest-market \
        prefect-server prefect-worker pipeline pipeline-deploy

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# ============================================================================
# HELP
# ============================================================================

help: ## Show this help message
	@echo ""
	@echo "$(BLUE)Geopolitical Market Tracker$(NC)"
	@echo "=============================="
	@echo ""
	@echo "$(GREEN)Docker Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(up|down|start|stop|restart|logs|build|clean|migrate)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Development Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(dev-)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Data Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(ingest-)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Prefect Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '(prefect-|pipeline)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Other Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -vE '(up|down|start|stop|restart|logs|build|clean|migrate|dev-|ingest-|prefect-|pipeline)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""

# ============================================================================
# DOCKER - FULL STACK
# ============================================================================

up: ## Start all services (database, API, dashboard)
	@echo "$(GREEN)Starting all services...$(NC)"
	docker compose up -d
	@echo ""
	@echo "$(GREEN)Services started!$(NC)"
	@echo "  Dashboard: http://localhost:8501"
	@echo "  API Docs:  http://localhost:8000/docs"
	@echo "  Database:  localhost:5432"

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

logs-dashboard: ## View dashboard logs only
	docker compose logs -f dashboard

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

# ============================================================================
# LOCAL DEVELOPMENT (without Docker)
# ============================================================================

dev-api: ## Run API locally (requires venv and running DB)
	@echo "$(BLUE)Starting API in development mode...$(NC)"
	. venv/bin/activate && uvicorn src.api.main:app --reload --port 8000

dev-dashboard: ## Run dashboard locally (direct DB mode)
	@echo "$(BLUE)Starting dashboard in development mode...$(NC)"
	. venv/bin/activate && streamlit run dashboard/app.py

dev-dashboard-api: ## Run dashboard locally (API mode)
	@echo "$(BLUE)Starting dashboard in API mode...$(NC)"
	. venv/bin/activate && USE_API=true API_URL=http://localhost:8000 streamlit run dashboard/app.py

dev: ## Run both API and dashboard locally (requires 2 terminals)
	@echo "$(YELLOW)Run these commands in separate terminals:$(NC)"
	@echo ""
	@echo "  Terminal 1 (API):       make dev-api"
	@echo "  Terminal 2 (Dashboard): make dev-dashboard"
	@echo ""

# ============================================================================
# DATA INGESTION
# ============================================================================

ingest-events: ## Ingest GDELT events (last 7 days)
	@echo "$(BLUE)Ingesting GDELT events...$(NC)"
	. venv/bin/activate && python -c "\
from datetime import date, timedelta; \
from src.ingestion.gdelt import GDELTIngestion; \
g = GDELTIngestion(); \
g.ingest_date_range(date.today() - timedelta(days=7), date.today() - timedelta(days=1)); \
"
	@echo "$(GREEN)Event ingestion complete.$(NC)"

ingest-market: ## Ingest market data (last 30 days)
	@echo "$(BLUE)Ingesting market data...$(NC)"
	. venv/bin/activate && python -c "\
from datetime import date, timedelta; \
from src.ingestion.market_data import MarketDataIngestion; \
m = MarketDataIngestion(); \
m.ingest_all_symbols(date.today() - timedelta(days=30), date.today()); \
"
	@echo "$(GREEN)Market data ingestion complete.$(NC)"

ingest-all: ingest-events ingest-market ## Ingest both events and market data

# ============================================================================
# TESTING & LINTING
# ============================================================================

test: ## Run tests
	@echo "$(BLUE)Running tests...$(NC)"
	. venv/bin/activate && pytest tests/ -v

lint: ## Run linters (black, flake8)
	@echo "$(BLUE)Running linters...$(NC)"
	. venv/bin/activate && black --check src/ dashboard/
	. venv/bin/activate && flake8 src/ dashboard/

format: ## Format code with black
	@echo "$(BLUE)Formatting code...$(NC)"
	. venv/bin/activate && black src/ dashboard/
	@echo "$(GREEN)Formatting complete.$(NC)"

# ============================================================================
# SETUP
# ============================================================================

install: ## Install Python dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)Dependencies installed.$(NC)"

setup: ## Full setup: install deps, start DB, run migrations
	@echo "$(BLUE)Running full setup...$(NC)"
	$(MAKE) install
	$(MAKE) up-db
	@echo "$(YELLOW)Waiting for database to be ready...$(NC)"
	sleep 5
	. venv/bin/activate && alembic upgrade head
	@echo "$(GREEN)Setup complete! Run 'make up' to start all services.$(NC)"

# ============================================================================
# PREFECT ORCHESTRATION (Local - without Docker)
# ============================================================================

prefect-server: ## Start Prefect server locally (UI at http://localhost:4200)
	@echo "$(BLUE)Starting Prefect server...$(NC)"
	@echo "$(GREEN)Prefect UI will be available at: http://localhost:4200$(NC)"
	. venv/bin/activate && prefect server start

prefect-worker: ## Start Prefect worker locally
	@echo "$(BLUE)Starting Prefect worker...$(NC)"
	. venv/bin/activate && prefect worker start -p 'default-agent-pool'

pipeline: ## Run the daily pipeline locally
	@echo "$(BLUE)Running daily pipeline...$(NC)"
	. venv/bin/activate && python flows/daily_pipeline.py
	@echo "$(GREEN)Pipeline complete.$(NC)"

pipeline-deploy: ## Create Prefect deployments locally
	@echo "$(BLUE)Creating Prefect deployments...$(NC)"
	. venv/bin/activate && python flows/daily_pipeline.py --deploy
	. venv/bin/activate && python flows/ingest_events.py --deploy
	. venv/bin/activate && python flows/ingest_market.py --deploy
	. venv/bin/activate && python flows/run_analysis.py --deploy
	@echo "$(GREEN)Deployments created!$(NC)"
	@echo ""
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Start Prefect server:  make prefect-server"
	@echo "  2. Start worker:          make prefect-worker"
	@echo "  3. View UI:               http://localhost:4200"

pipeline-events: ## Run only event ingestion flow locally
	@echo "$(BLUE)Running event ingestion flow...$(NC)"
	. venv/bin/activate && python flows/ingest_events.py --days 7

pipeline-market: ## Run only market data ingestion flow locally
	@echo "$(BLUE)Running market data ingestion flow...$(NC)"
	. venv/bin/activate && python flows/ingest_market.py --days 30

pipeline-analysis: ## Run only analysis flow locally
	@echo "$(BLUE)Running analysis flow...$(NC)"
	. venv/bin/activate && python flows/run_analysis.py --days 30

# ============================================================================
# PREFECT ORCHESTRATION (Docker - containerized)
# ============================================================================

up-prefect: ## Start Prefect server and worker in Docker
	@echo "$(BLUE)Starting Prefect services...$(NC)"
	docker compose --profile prefect up -d
	@echo ""
	@echo "$(GREEN)Prefect services started!$(NC)"
	@echo "  Prefect UI: http://localhost:4200"
	@echo ""
	@echo "$(YELLOW)To create scheduled deployments, run:$(NC)"
	@echo "  make prefect-deploy-docker"

down-prefect: ## Stop Prefect services
	@echo "$(YELLOW)Stopping Prefect services...$(NC)"
	docker compose --profile prefect down
	@echo "$(GREEN)Prefect services stopped.$(NC)"

prefect-deploy-docker: ## Create Prefect deployments in Docker
	@echo "$(BLUE)Creating Prefect deployments...$(NC)"
	docker compose --profile prefect-setup up prefect-deploy
	@echo "$(GREEN)Deployments created!$(NC)"

pipeline-docker: ## Run daily pipeline in Docker (one-time)
	@echo "$(BLUE)Running daily pipeline in Docker...$(NC)"
	docker compose --profile pipeline up pipeline
	@echo "$(GREEN)Pipeline complete.$(NC)"

logs-prefect: ## View Prefect logs
	docker compose --profile prefect logs -f prefect-server prefect-worker

up-all: ## Start ALL services including Prefect
	@echo "$(GREEN)Starting all services...$(NC)"
	docker compose --profile prefect up -d
	@echo ""
	@echo "$(GREEN)All services started!$(NC)"
	@echo "  Dashboard:  http://localhost:8501"
	@echo "  API Docs:   http://localhost:8000/docs"
	@echo "  Prefect UI: http://localhost:4200"
