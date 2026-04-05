"""
Model Trainer — trains all 6 models and logs results to MLflow.

Each model is trained on the same dataset (from ml_features.py), evaluated
on the same test set, and logged to MLflow with params, metrics, and artifacts.

The trainer produces a comparison table and identifies the best model by AUC-ROC.

USAGE:
------
    from src.training.trainer import ModelTrainer

    trainer = ModelTrainer(symbols=["SPY", "CL=F", "GC=F"])
    results = trainer.run_all()
    print(results["comparison"])  # DataFrame comparing all models
    print(results["best_model"])  # Name of the best model
"""

import json
import logging
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import xgboost as xgb
import lightgbm as lgb

from src.config.settings import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from src.analysis.ml_features import MLFeaturePipeline

logger = logging.getLogger(__name__)

# Name under which the winning model is registered in the MLflow Model Registry
CHAMPION_MODEL_NAME = "gmip-event-impact-champion"
CHAMPION_ALIAS = "champion"

# Symbols to train on — diversified across asset classes
DEFAULT_SYMBOLS = [
    "SPY", "CL=F", "GC=F", "^VIX", "TLT", "EEM",
    "EURUSD=X", "NG=F",
]


class ModelTrainer:
    """Trains and evaluates all models, logging everything to MLflow."""

    def __init__(
        self,
        symbols: list[str] = None,
        start_date: date = None,
        end_date: date = None,
    ):
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.start_date = start_date or date(2016, 1, 1)
        self.end_date = end_date or date.today()
        self.pipeline = MLFeaturePipeline()

        # Preprocessing state — populated during run_all() before training
        self._scaler: StandardScaler | None = None
        self._feature_names: list[str] = []

        # Set up MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    def run_all(self) -> dict:
        """Train all 6 models and return comparison results.

        Returns dict with:
            comparison: DataFrame of model metrics
            best_model: name of the best model by AUC-ROC
            results: dict of model_name -> metrics dict
            run_ids: dict of model_name -> MLflow run_id
            champion: name of the model registered as champion
        """
        logger.info(f"Building datasets for {len(self.symbols)} symbols...")

        # Build datasets
        flat_data = self.pipeline.build_flat_dataset(
            self.symbols,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        seq_data = self.pipeline.build_sequence_dataset(
            self.symbols,
            start_date=self.start_date,
            end_date=self.end_date,
        )

        logger.info(
            f"Flat: {flat_data['X_train'].shape[0]} train, "
            f"{flat_data['X_val'].shape[0]} val, "
            f"{flat_data['X_test'].shape[0]} test"
        )
        logger.info(
            f"Sequences: {seq_data['X_train'].shape[0]} train, "
            f"window={seq_data['window_size']}"
        )

        # Scale flat features (shared across flat models)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(flat_data["X_train"])
        X_val_scaled = scaler.transform(flat_data["X_val"])
        X_test_scaled = scaler.transform(flat_data["X_test"])

        # Store the scaler + feature names so inference uses the same pipeline
        self._scaler = scaler
        self._feature_names = flat_data["feature_names"]

        results = {}
        run_ids = {}

        # --- 1. Logistic Regression (baseline) ---
        results["logistic_regression"], run_ids["logistic_regression"] = self._train_sklearn(
            name="logistic_regression",
            model=LogisticRegression(max_iter=1000, random_state=42),
            X_train=X_train_scaled,
            X_val=X_val_scaled,
            X_test=X_test_scaled,
            y_train=flat_data["y_train"],
            y_val=flat_data["y_val"],
            y_test=flat_data["y_test"],
            feature_names=flat_data["feature_names"],
            params={"model_type": "logistic_regression", "max_iter": 1000},
        )

        # --- 2. Random Forest ---
        results["random_forest"], run_ids["random_forest"] = self._train_sklearn(
            name="random_forest",
            model=RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1,
            ),
            X_train=X_train_scaled,
            X_val=X_val_scaled,
            X_test=X_test_scaled,
            y_train=flat_data["y_train"],
            y_val=flat_data["y_val"],
            y_test=flat_data["y_test"],
            feature_names=flat_data["feature_names"],
            params={
                "model_type": "random_forest",
                "n_estimators": 200,
                "max_depth": 10,
            },
        )

        # --- 3. XGBoost ---
        results["xgboost"], run_ids["xgboost"] = self._train_xgboost(
            X_train=X_train_scaled,
            X_val=X_val_scaled,
            X_test=X_test_scaled,
            y_train=flat_data["y_train"],
            y_val=flat_data["y_val"],
            y_test=flat_data["y_test"],
            feature_names=flat_data["feature_names"],
        )

        # --- 4. LightGBM ---
        results["lightgbm"], run_ids["lightgbm"] = self._train_lightgbm(
            X_train=X_train_scaled,
            X_val=X_val_scaled,
            X_test=X_test_scaled,
            y_train=flat_data["y_train"],
            y_val=flat_data["y_val"],
            y_test=flat_data["y_test"],
            feature_names=flat_data["feature_names"],
        )

        # --- 5. MLP ---
        results["mlp"], run_ids["mlp"] = self._train_mlp(
            X_train=X_train_scaled,
            X_val=X_val_scaled,
            X_test=X_test_scaled,
            y_train=flat_data["y_train"],
            y_val=flat_data["y_val"],
            y_test=flat_data["y_test"],
            feature_names=flat_data["feature_names"],
        )

        # --- 6. LSTM ---
        results["lstm"], run_ids["lstm"] = self._train_lstm(
            seq_data=seq_data,
        )

        # Build comparison table
        comparison = pd.DataFrame(results).T
        comparison = comparison.sort_values("test_auc", ascending=False)

        best_model = comparison.index[0]
        best_auc = comparison.loc[best_model, "test_auc"]
        logger.info(f"Best model: {best_model} (AUC={best_auc:.4f})")

        # Register the champion in MLflow Model Registry
        # Only tree / sklearn models are registered — PyTorch models need
        # different loading code and aren't wired up yet
        champion_registered = None
        if best_model in run_ids and best_model not in {"mlp", "lstm"}:
            champion_registered = self._register_champion(
                run_id=run_ids[best_model],
                model_name=best_model,
                test_auc=best_auc,
            )
        else:
            logger.info(
                f"Best model '{best_model}' is a PyTorch model. Champion registry "
                f"currently only supports sklearn-compatible models — skipping."
            )
            # Fall back to the best sklearn-compatible model
            sklearn_models = ["xgboost", "lightgbm", "random_forest", "logistic_regression"]
            for sk in sklearn_models:
                if sk in comparison.index:
                    fallback_auc = comparison.loc[sk, "test_auc"]
                    logger.info(f"Registering {sk} (AUC={fallback_auc:.4f}) as champion instead")
                    champion_registered = self._register_champion(
                        run_id=run_ids[sk],
                        model_name=sk,
                        test_auc=fallback_auc,
                    )
                    break

        return {
            "comparison": comparison,
            "best_model": best_model,
            "results": results,
            "run_ids": run_ids,
            "champion": champion_registered,
        }

    # ------------------------------------------------------------------
    # MLflow artifacts and model registry
    # ------------------------------------------------------------------

    def _log_preprocessing_artifacts(self, feature_names: list[str]) -> None:
        """Log the StandardScaler and feature names so predictions use the same pipeline.

        Called inside every active MLflow run. The scaler and feature list are
        saved as artifacts so the model loader can reconstruct the exact
        preprocessing the model was trained with.
        """
        if self._scaler is None:
            return

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # Save the fitted scaler
            scaler_path = tmp_path / "scaler.joblib"
            joblib.dump(self._scaler, scaler_path)
            mlflow.log_artifact(str(scaler_path), artifact_path="preprocessing")

            # Save feature names
            features_path = tmp_path / "feature_names.json"
            features_path.write_text(json.dumps(feature_names))
            mlflow.log_artifact(str(features_path), artifact_path="preprocessing")

    def _register_champion(
        self, run_id: str, model_name: str, test_auc: float,
    ) -> dict | None:
        """Register the best model in MLflow Model Registry under the champion alias.

        If the current champion has a higher test AUC, the new model is still
        registered as a new version, but does NOT get the champion alias.
        This implements simple champion/challenger promotion.
        """
        try:
            client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

            # Register the new model version
            model_uri = f"runs:/{run_id}/model"
            registered = mlflow.register_model(
                model_uri=model_uri,
                name=CHAMPION_MODEL_NAME,
                tags={
                    "model_type": model_name,
                    "test_auc": f"{test_auc:.4f}",
                },
            )
            new_version = registered.version
            logger.info(
                f"Registered {model_name} as {CHAMPION_MODEL_NAME} v{new_version} "
                f"(AUC={test_auc:.4f})"
            )

            # Check if an existing champion exists
            try:
                current_champion = client.get_model_version_by_alias(
                    CHAMPION_MODEL_NAME, CHAMPION_ALIAS,
                )
                current_auc = float(current_champion.tags.get("test_auc", 0))
                logger.info(
                    f"Current champion: v{current_champion.version} (AUC={current_auc:.4f})"
                )
                if test_auc <= current_auc:
                    logger.info(
                        f"New model AUC {test_auc:.4f} does not beat current "
                        f"champion {current_auc:.4f}. Keeping existing champion."
                    )
                    return {
                        "model_name": current_champion.tags.get("model_type"),
                        "version": current_champion.version,
                        "test_auc": current_auc,
                        "promoted": False,
                    }
            except Exception:
                # No existing champion — this is the first training run
                logger.info("No existing champion — promoting this model as first champion.")

            # Promote to champion
            client.set_registered_model_alias(
                name=CHAMPION_MODEL_NAME,
                alias=CHAMPION_ALIAS,
                version=new_version,
            )
            logger.info(f"Promoted v{new_version} ({model_name}) to champion")

            return {
                "model_name": model_name,
                "version": new_version,
                "test_auc": test_auc,
                "promoted": True,
            }
        except Exception as e:
            logger.error(f"Champion registration failed: {e}")
            return None

    # ------------------------------------------------------------------
    # sklearn models (LogReg, RF)
    # ------------------------------------------------------------------

    def _train_sklearn(
        self,
        name: str,
        model,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        feature_names: list[str],
        params: dict,
    ) -> tuple[dict, str]:
        """Train a sklearn classifier and log to MLflow.

        Returns (metrics, run_id).
        """
        logger.info(f"Training {name}...")

        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params(params)
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("symbols", ",".join(self.symbols))

            model.fit(X_train, y_train)

            metrics = self._evaluate_and_log(
                model, X_train, X_val, X_test,
                y_train, y_val, y_test, feature_names,
            )

            mlflow.sklearn.log_model(model, "model")
            self._log_preprocessing_artifacts(feature_names)

        return metrics, run.info.run_id

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------

    def _train_xgboost(
        self, X_train, X_val, X_test, y_train, y_val, y_test, feature_names,
    ) -> tuple[dict, str]:
        """Train XGBoost with early stopping on validation set."""
        logger.info("Training xgboost...")

        params = {
            "model_type": "xgboost",
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }

        with mlflow.start_run(run_name="xgboost") as run:
            mlflow.log_params(params)
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("symbols", ",".join(self.symbols))

            model = xgb.XGBClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                random_state=42,
                eval_metric="logloss",
                early_stopping_rounds=20,
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            metrics = self._evaluate_and_log(
                model, X_train, X_val, X_test,
                y_train, y_val, y_test, feature_names,
            )

            mlflow.xgboost.log_model(model, "model")
            self._log_preprocessing_artifacts(feature_names)

        return metrics, run.info.run_id

    # ------------------------------------------------------------------
    # LightGBM
    # ------------------------------------------------------------------

    def _train_lightgbm(
        self, X_train, X_val, X_test, y_train, y_val, y_test, feature_names,
    ) -> tuple[dict, str]:
        """Train LightGBM with early stopping on validation set."""
        logger.info("Training lightgbm...")

        params = {
            "model_type": "lightgbm",
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "num_leaves": 31,
        }

        with mlflow.start_run(run_name="lightgbm") as run:
            mlflow.log_params(params)
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("symbols", ",".join(self.symbols))

            model = lgb.LGBMClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                num_leaves=params["num_leaves"],
                random_state=42,
                verbose=-1,
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(20, verbose=False)],
            )

            metrics = self._evaluate_and_log(
                model, X_train, X_val, X_test,
                y_train, y_val, y_test, feature_names,
            )

            mlflow.lightgbm.log_model(model, "model")
            self._log_preprocessing_artifacts(feature_names)

        return metrics, run.info.run_id

    # ------------------------------------------------------------------
    # MLP (PyTorch)
    # ------------------------------------------------------------------

    def _train_mlp(
        self, X_train, X_val, X_test, y_train, y_val, y_test, feature_names,
    ) -> tuple[dict, str]:
        """Train a simple MLP in PyTorch."""
        logger.info("Training mlp...")
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        params = {
            "model_type": "mlp",
            "hidden_sizes": "128,64",
            "dropout": 0.3,
            "learning_rate": 0.001,
            "epochs": 50,
            "batch_size": 256,
        }

        class MLP(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 1),
                )

            def forward(self, x):
                return self.net(x)

        with mlflow.start_run(run_name="mlp") as run:
            mlflow.log_params(params)
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("symbols", ",".join(self.symbols))

            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

            model = MLP(X_train.shape[1]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.BCEWithLogitsLoss()

            # DataLoaders
            train_ds = TensorDataset(
                torch.FloatTensor(X_train), torch.FloatTensor(y_train),
            )
            train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

            # Training loop
            model.train()
            for epoch in range(50):
                epoch_loss = 0.0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    logits = model(X_batch).squeeze()
                    loss = criterion(logits, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                if (epoch + 1) % 10 == 0:
                    mlflow.log_metric("train_loss", epoch_loss / len(train_loader), step=epoch)

            # Evaluate
            model.eval()
            metrics = self._evaluate_pytorch(
                model, device, X_train, X_val, X_test,
                y_train, y_val, y_test, feature_names,
            )
            self._log_preprocessing_artifacts(feature_names)

        return metrics, run.info.run_id

    # ------------------------------------------------------------------
    # LSTM (PyTorch)
    # ------------------------------------------------------------------

    def _train_lstm(self, seq_data: dict) -> tuple[dict, str]:
        """Train an LSTM on windowed sequences."""
        logger.info("Training lstm...")
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        params = {
            "model_type": "lstm",
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.3,
            "learning_rate": 0.001,
            "epochs": 50,
            "batch_size": 128,
            "window_size": seq_data["window_size"],
        }

        class LSTMClassifier(nn.Module):
            def __init__(self, input_dim, hidden_size=64, num_layers=2):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=0.3,
                )
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                # x shape: (batch, seq_len, features)
                lstm_out, _ = self.lstm(x)
                # Take the last timestep's output
                last_hidden = lstm_out[:, -1, :]
                return self.fc(last_hidden)

        with mlflow.start_run(run_name="lstm") as run:
            mlflow.log_params(params)
            mlflow.log_param("n_train", len(seq_data["X_train"]))
            mlflow.log_param("n_features", seq_data["X_train"].shape[2])
            mlflow.log_param("symbols", ",".join(self.symbols))

            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

            input_dim = seq_data["X_train"].shape[2]
            model = LSTMClassifier(input_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.BCEWithLogitsLoss()

            # DataLoaders
            train_ds = TensorDataset(
                torch.FloatTensor(seq_data["X_train"]),
                torch.FloatTensor(seq_data["y_train"]),
            )
            train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

            # Training loop
            model.train()
            for epoch in range(50):
                epoch_loss = 0.0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    logits = model(X_batch).squeeze()
                    loss = criterion(logits, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                if (epoch + 1) % 10 == 0:
                    mlflow.log_metric("train_loss", epoch_loss / len(train_loader), step=epoch)

            # Evaluate
            model.eval()
            metrics = self._evaluate_pytorch_seq(
                model, device, seq_data,
            )

        return metrics, run.info.run_id

    # ------------------------------------------------------------------
    # Shared evaluation helpers
    # ------------------------------------------------------------------

    def _evaluate_and_log(
        self, model, X_train, X_val, X_test,
        y_train, y_val, y_test, feature_names,
    ) -> dict:
        """Evaluate a sklearn-compatible model and log metrics to MLflow."""
        metrics = {}

        for split_name, X, y in [
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred

            acc = accuracy_score(y, y_pred)
            prec = precision_score(y, y_pred, zero_division=0)
            rec = recall_score(y, y_pred, zero_division=0)
            f1 = f1_score(y, y_pred, zero_division=0)
            auc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0.0

            metrics[f"{split_name}_accuracy"] = acc
            metrics[f"{split_name}_precision"] = prec
            metrics[f"{split_name}_recall"] = rec
            metrics[f"{split_name}_f1"] = f1
            metrics[f"{split_name}_auc"] = auc

            mlflow.log_metric(f"{split_name}_accuracy", acc)
            mlflow.log_metric(f"{split_name}_precision", prec)
            mlflow.log_metric(f"{split_name}_recall", rec)
            mlflow.log_metric(f"{split_name}_f1", f1)
            mlflow.log_metric(f"{split_name}_auc", auc)

        # Log feature importance if available
        if hasattr(model, "feature_importances_"):
            importance = dict(zip(feature_names, model.feature_importances_))
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            for feat, imp in sorted_imp[:10]:
                mlflow.log_metric(f"importance_{feat}", imp)

        # Log classification report as artifact
        report = classification_report(y_test, model.predict(X_test))
        mlflow.log_text(report, "classification_report.txt")

        logger.info(
            f"  test: acc={metrics['test_accuracy']:.4f}, "
            f"auc={metrics['test_auc']:.4f}, "
            f"f1={metrics['test_f1']:.4f}"
        )

        return metrics

    def _evaluate_pytorch(
        self, model, device, X_train, X_val, X_test,
        y_train, y_val, y_test, feature_names,
    ) -> dict:
        """Evaluate a PyTorch model on flat features."""
        import torch

        metrics = {}

        for split_name, X, y in [
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            with torch.no_grad():
                X_t = torch.FloatTensor(X).to(device)
                logits = model(X_t).squeeze().cpu().numpy()
                y_prob = 1 / (1 + np.exp(-logits))  # sigmoid
                y_pred = (y_prob >= 0.5).astype(int)

            acc = accuracy_score(y, y_pred)
            prec = precision_score(y, y_pred, zero_division=0)
            rec = recall_score(y, y_pred, zero_division=0)
            f1 = f1_score(y, y_pred, zero_division=0)
            auc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0.0

            metrics[f"{split_name}_accuracy"] = acc
            metrics[f"{split_name}_precision"] = prec
            metrics[f"{split_name}_recall"] = rec
            metrics[f"{split_name}_f1"] = f1
            metrics[f"{split_name}_auc"] = auc

            mlflow.log_metric(f"{split_name}_accuracy", acc)
            mlflow.log_metric(f"{split_name}_precision", prec)
            mlflow.log_metric(f"{split_name}_recall", rec)
            mlflow.log_metric(f"{split_name}_f1", f1)
            mlflow.log_metric(f"{split_name}_auc", auc)

        logger.info(
            f"  test: acc={metrics['test_accuracy']:.4f}, "
            f"auc={metrics['test_auc']:.4f}, "
            f"f1={metrics['test_f1']:.4f}"
        )

        return metrics

    def _evaluate_pytorch_seq(self, model, device, seq_data: dict) -> dict:
        """Evaluate a PyTorch model on sequence data."""
        import torch

        metrics = {}

        for split_name in ["train", "val", "test"]:
            X = seq_data[f"X_{split_name}"]
            y = seq_data[f"y_{split_name}"]

            with torch.no_grad():
                X_t = torch.FloatTensor(X).to(device)
                logits = model(X_t).squeeze().cpu().numpy()
                y_prob = 1 / (1 + np.exp(-logits))  # sigmoid
                y_pred = (y_prob >= 0.5).astype(int)

            acc = accuracy_score(y, y_pred)
            prec = precision_score(y, y_pred, zero_division=0)
            rec = recall_score(y, y_pred, zero_division=0)
            f1 = f1_score(y, y_pred, zero_division=0)
            auc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0.0

            metrics[f"{split_name}_accuracy"] = acc
            metrics[f"{split_name}_precision"] = prec
            metrics[f"{split_name}_recall"] = rec
            metrics[f"{split_name}_f1"] = f1
            metrics[f"{split_name}_auc"] = auc

            mlflow.log_metric(f"{split_name}_accuracy", acc)
            mlflow.log_metric(f"{split_name}_precision", prec)
            mlflow.log_metric(f"{split_name}_recall", rec)
            mlflow.log_metric(f"{split_name}_f1", f1)
            mlflow.log_metric(f"{split_name}_auc", auc)

        logger.info(
            f"  test: acc={metrics['test_accuracy']:.4f}, "
            f"auc={metrics['test_auc']:.4f}, "
            f"f1={metrics['test_f1']:.4f}"
        )

        return metrics
