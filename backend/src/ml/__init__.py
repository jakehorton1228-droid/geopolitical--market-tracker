"""
ML Runtime Module — loads and serves trained models from MLflow.

The training module (src/training/) produces models and registers the
winner as champion in MLflow Model Registry. This module is the runtime
counterpart — it loads the champion model and runs predictions against it.
"""
