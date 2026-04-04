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

import logging
from datetime import date

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
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

        # Set up MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    def run_all(self) -> dict:
        """Train all 6 models and return comparison results.

        Returns dict with:
            comparison: DataFrame of model metrics
            best_model: name of the best model by AUC-ROC
            results: dict of model_name -> metrics dict
        """
        logger.info(f"Building datasets for {len(self.symbols)} symbols...")

        # Build datasets
        flat_data = self.pipeline.build_flat_dataset(
            self.symbols, self.start_date, self.end_date
        )
        seq_data = self.pipeline.build_sequence_dataset(
            self.symbols, self.start_date, self.end_date
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

        results = {}

        # --- 1. Logistic Regression (baseline) ---
        results["logistic_regression"] = self._train_sklearn(
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
        results["random_forest"] = self._train_sklearn(
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
        results["xgboost"] = self._train_xgboost(
            X_train=X_train_scaled,
            X_val=X_val_scaled,
            X_test=X_test_scaled,
            y_train=flat_data["y_train"],
            y_val=flat_data["y_val"],
            y_test=flat_data["y_test"],
            feature_names=flat_data["feature_names"],
        )

        # --- 4. LightGBM ---
        results["lightgbm"] = self._train_lightgbm(
            X_train=X_train_scaled,
            X_val=X_val_scaled,
            X_test=X_test_scaled,
            y_train=flat_data["y_train"],
            y_val=flat_data["y_val"],
            y_test=flat_data["y_test"],
            feature_names=flat_data["feature_names"],
        )

        # --- 5. MLP ---
        results["mlp"] = self._train_mlp(
            X_train=X_train_scaled,
            X_val=X_val_scaled,
            X_test=X_test_scaled,
            y_train=flat_data["y_train"],
            y_val=flat_data["y_val"],
            y_test=flat_data["y_test"],
            feature_names=flat_data["feature_names"],
        )

        # --- 6. LSTM ---
        results["lstm"] = self._train_lstm(
            seq_data=seq_data,
        )

        # Build comparison table
        comparison = pd.DataFrame(results).T
        comparison = comparison.sort_values("test_auc", ascending=False)

        best_model = comparison.index[0]
        logger.info(f"Best model: {best_model} (AUC={comparison.loc[best_model, 'test_auc']:.4f})")

        return {
            "comparison": comparison,
            "best_model": best_model,
            "results": results,
        }

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
    ) -> dict:
        """Train a sklearn classifier and log to MLflow."""
        logger.info(f"Training {name}...")

        with mlflow.start_run(run_name=name):
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

        return metrics

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------

    def _train_xgboost(
        self, X_train, X_val, X_test, y_train, y_val, y_test, feature_names,
    ) -> dict:
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

        with mlflow.start_run(run_name="xgboost"):
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

        return metrics

    # ------------------------------------------------------------------
    # LightGBM
    # ------------------------------------------------------------------

    def _train_lightgbm(
        self, X_train, X_val, X_test, y_train, y_val, y_test, feature_names,
    ) -> dict:
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

        with mlflow.start_run(run_name="lightgbm"):
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

        return metrics

    # ------------------------------------------------------------------
    # MLP (PyTorch)
    # ------------------------------------------------------------------

    def _train_mlp(
        self, X_train, X_val, X_test, y_train, y_val, y_test, feature_names,
    ) -> dict:
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

        with mlflow.start_run(run_name="mlp"):
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

        return metrics

    # ------------------------------------------------------------------
    # LSTM (PyTorch)
    # ------------------------------------------------------------------

    def _train_lstm(self, seq_data: dict) -> dict:
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

        with mlflow.start_run(run_name="lstm"):
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

        return metrics

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
