"""
Prefect flow for ML model training.

Runs the full training pipeline:
1. Build feature datasets (flat + windowed sequences)
2. Train all 6 models (LogReg, RF, XGBoost, LightGBM, MLP, LSTM)
3. Log everything to MLflow (params, metrics, artifacts)
4. Report the best model by AUC-ROC

Designed to run weekly via Prefect schedule, or manually on demand.
MLflow UI at http://localhost:5000 shows all experiment results.
"""

from prefect import flow, task, get_run_logger


@task(name="train-all-models", retries=0, log_prints=True, timeout_seconds=1800)
def train_models(symbols: list[str] = None) -> dict:
    """Train all 6 models and log results to MLflow."""
    logger = get_run_logger()

    from src.training.trainer import ModelTrainer, DEFAULT_SYMBOLS

    symbols = symbols or DEFAULT_SYMBOLS
    logger.info(f"Starting model training for {len(symbols)} symbols")

    trainer = ModelTrainer(symbols=symbols)
    results = trainer.run_all()

    comparison = results["comparison"]
    best = results["best_model"]

    logger.info(f"Training complete. Best model: {best}")
    logger.info(f"\n{comparison.to_string()}")

    return {
        "best_model": best,
        "best_auc": float(comparison.loc[best, "test_auc"]),
        "n_models": len(comparison),
    }


@flow(name="weekly-model-training", log_prints=True)
def weekly_training(symbols: list[str] = None):
    """
    Weekly model training pipeline.

    Trains all models on the latest data and logs to MLflow.
    Run manually or schedule via Prefect.
    """
    logger = get_run_logger()
    logger.info("Starting weekly model training pipeline")

    result = train_models(symbols=symbols)

    logger.info(
        f"Pipeline complete. Best model: {result['best_model']} "
        f"(AUC={result['best_auc']:.4f})"
    )

    return result


if __name__ == "__main__":
    weekly_training()
