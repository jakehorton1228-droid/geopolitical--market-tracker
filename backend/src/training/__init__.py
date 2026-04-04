"""
Training Module — trains and evaluates ML models for event impact classification.

Trains 6 models on the same dataset, logs all runs to MLflow, and selects
the best performer based on AUC-ROC on the held-out test set.

Models:
  Flat features (27 features per sample):
    1. Logistic Regression (baseline)
    2. Random Forest
    3. XGBoost
    4. LightGBM
    5. MLP (PyTorch)

  Windowed sequences (30 days x 16 features):
    6. LSTM (PyTorch)

All models share a time-series aware train/val/test split via ml_features.py.
"""
