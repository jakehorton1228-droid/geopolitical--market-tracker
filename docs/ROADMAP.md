# GMIP Roadmap — Remaining Work

## Completed (2026-04-03)
- [x] Swap Anthropic API for local Ollama/Llama 3 inference
- [x] Make LangGraph supervisor deterministic (state-based routing)
- [x] Make collection/analysis nodes deterministic (no LLM tool calling)
- [x] Only dissemination node uses LLM (synthesis layer)
- [x] Remove single-agent mode, consolidate to one LangGraph pipeline
- [x] Add Ollama to Docker Compose
- [x] Create shared synthesis module (analysis/synthesis.py)
- [x] Clean up all Anthropic references across codebase + README
- [x] Add architecture diagrams (docs/*.mmd)

## Completed (2026-04-04)
- [x] Build ML feature pipeline (analysis/ml_features.py)
  - Flat features (27 columns) for tree models + MLP
  - Windowed sequences (30 x 16) for CNN/LSTM
  - Time-series aware train/val/test split (70/15/15)
  - Sentiment, volatility, momentum, temporal, event velocity features
- [x] Update all analysis module docstrings and comments
- [x] Add MLflow tracking server to Docker Compose (port 5000)
- [x] Add XGBoost, LightGBM, MLflow to requirements.txt
- [x] Add MLflow config to settings.py
- [x] Build model training module (training/trainer.py)
  - 6 models: Logistic Regression, Random Forest, XGBoost, LightGBM, MLP, LSTM
  - All logged to MLflow (params, metrics, artifacts, feature importance)
  - Comparison table, best model selection by AUC-ROC
  - MPS acceleration for PyTorch models on Apple Silicon
- [x] Create training Prefect flow (flows/training_flow.py)

## Remaining Work

### 1. LangSmith Tracing
- Instrument LangGraph pipeline with LangSmith observability
- Traces on every node execution, state transition, LLM call
- ~1 day effort

### 2. Model Evaluation Dashboard
- New frontend page or section showing:
  - Classification metrics (precision, recall, F1, AUC-ROC)
  - Feature importance plots
  - Calibration curves
  - Model comparison charts
- Pull data from MLflow API or add API endpoints

### 3. Fine-tune Llama 3 8B (stretch goal)
- Fine-tune on domain-specific intelligence assessments
- Training data: generate examples using Claude Max subscription
- Method: QLoRA via mlx-lm (Apple Silicon optimized)
- ~10-14GB VRAM needed, fits in 48GB unified memory easily

### 4. Polish and Portfolio Artifacts
- Record Loom walkthrough (3-5 minutes)
- Create public GitHub showcase repo with architecture diagram + screenshots
- Clean up private repo (consistent style, docstrings, Docker instructions)

## Architecture

```
Data Ingestion (Prefect, 5 sources, deterministic)
       ↓
Feature Engineering (ml_features.py)
  → Flat: 27 features for tree models + MLP
  → Windowed: 30-day sequences for LSTM
       ↓
Model Training (training/trainer.py)
  → 6 models: LogReg, RF, XGBoost, LightGBM, MLP, LSTM
  → MLflow experiment tracking (port 5000)
  → Weekly retraining via Prefect
       ↓
Analysis Pipeline (deterministic)
  → Correlation, anomaly, pattern, event study, regression
  → FinBERT sentiment + MiniLM embeddings
       ↓
LangGraph Intelligence Pipeline
  → Supervisor (deterministic state routing)
  → Collection (deterministic query parsing)
  → Analysis (deterministic tool execution)
  → Dissemination (Llama 3 8B via Ollama)
       ↓
Dashboard (React, 8 pages) + API (FastAPI, 11 routers)
  → Docker Compose: 7 services (db, api, frontend, prefect x2, ollama, mlflow)
```

## Resume Target

```
Geopolitical Market Intelligence Platform
- Trained and compared 6 ML models (XGBoost, LightGBM, RF, LogReg,
  MLP, LSTM) with MLflow experiment tracking and time-series cross-validation
- Built ML feature pipeline: 27 flat features + 30-day windowed sequences
  from 5 data sources (GDELT, FRED, Yahoo Finance, Polymarket, RSS)
- LangGraph multi-agent pipeline with deterministic data collection,
  statistical analysis, and local Llama 3 8B synthesis via Ollama
- RAG pipeline with pgvector similarity search for historical analogs
- NLP: FinBERT sentiment analysis + MiniLM embeddings
- Full-stack: FastAPI + React + PostgreSQL/pgvector + Docker Compose
- Automated: Prefect orchestration for daily ingestion, analysis, and
  weekly model retraining
```
