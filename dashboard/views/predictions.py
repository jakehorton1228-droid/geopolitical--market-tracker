"""
Predictions Page - ML Model Results.

Shows market direction predictions using trained Gradient Boosting and LSTM models.
"""

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
import numpy as np

from src.db.connection import get_session
from src.db.models import Event, MarketData
from src.config.constants import (
    SYMBOLS, get_all_symbols, get_symbol_info,
    CAMEO_CATEGORIES, EVENT_GROUPS, get_event_group
)
from sqlalchemy import func

# Try to import ML models (may not be installed)
try:
    from src.analysis.gradient_boost_classifier import GradientBoostClassifier
    GB_AVAILABLE = True
except ImportError:
    GB_AVAILABLE = False

try:
    from src.analysis.lstm_model import MarketLSTM, LSTMTrainer
    from src.analysis.sequence_dataset import MarketSequenceDataset
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False


def render():
    """Render the predictions page."""
    st.title("🎯 ML Market Predictions")
    st.markdown("Predict market direction using advanced ML models trained on geopolitical events.")

    # Show model availability
    col1, col2, col3 = st.columns(3)
    with col1:
        if GB_AVAILABLE:
            st.success("✅ Gradient Boosting Available")
        else:
            st.error("❌ Install: `pip install xgboost lightgbm`")
    with col2:
        if LSTM_AVAILABLE:
            st.success("✅ LSTM Available")
        else:
            st.error("❌ Install: `pip install torch`")
    with col3:
        st.info("📊 Models train on your data")

    # Model explanation
    with st.expander("How do these models work?", expanded=False):
        st.markdown("""
        ### Gradient Boosting (XGBoost / LightGBM)

        **How it works:**
        - Builds many decision trees sequentially
        - Each tree fixes the mistakes of previous trees
        - Combines all trees for final prediction

        **Strengths:**
        - Excellent for tabular data with mixed features
        - Fast training and inference
        - Built-in feature importance

        ---

        ### LSTM (Long Short-Term Memory)

        **How it works:**
        - Processes sequences of past days (e.g., last 20 days)
        - Has "memory cells" that remember important patterns
        - Learns temporal dependencies (what happened yesterday affects today)

        **Strengths:**
        - Captures sequential patterns
        - Models market "momentum" and trends
        - Can learn complex temporal relationships

        ---

        **Baseline:** Random guessing = 50% accuracy, 0.500 AUC

        Any model above 52-53% with consistent results is potentially valuable.
        """)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Make Prediction", "📊 Train & Evaluate", "📈 Feature Importance"])

    with tab1:
        render_prediction_interface()

    with tab2:
        render_training_interface()

    with tab3:
        render_feature_importance()


def render_prediction_interface():
    """Interactive interface to make predictions."""
    st.subheader("Predict Market Direction")

    if not GB_AVAILABLE:
        st.warning("Gradient Boosting not available. Install `xgboost` and `lightgbm`.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Event Characteristics**")

        goldstein_mean = st.slider(
            "Average Goldstein Scale",
            min_value=-10.0, max_value=10.0, value=-2.0, step=0.5,
            help="Average conflict/cooperation score for the day",
        )

        goldstein_min = st.slider(
            "Minimum Goldstein (worst event)",
            min_value=-10.0, max_value=10.0, value=-5.0, step=0.5,
        )

        goldstein_max = st.slider(
            "Maximum Goldstein (best event)",
            min_value=-10.0, max_value=10.0, value=2.0, step=0.5,
        )

        mentions_total = st.number_input(
            "Total Media Mentions",
            min_value=0, max_value=10000, value=500,
        )

    with col2:
        st.markdown("**Additional Features**")

        avg_tone = st.slider(
            "Average Tone",
            min_value=-10.0, max_value=10.0, value=-1.5, step=0.5,
        )

        conflict_count = st.number_input(
            "Number of Conflict Events",
            min_value=0, max_value=100, value=3,
        )

        cooperation_count = st.number_input(
            "Number of Cooperation Events",
            min_value=0, max_value=100, value=1,
        )

        st.markdown("**Target Market**")
        symbol_options = []
        for category, symbols in SYMBOLS.items():
            for symbol, name in symbols.items():
                symbol_options.append(f"{symbol} - {name}")

        selected = st.selectbox("Select Market", options=symbol_options, index=0)
        target_symbol = selected.split(" - ")[0]

    # Predict button
    st.markdown("---")
    if st.button("🔮 Predict with Gradient Boosting", type="primary", use_container_width=True):
        with st.spinner("Training model and making prediction..."):
            try:
                # Get date range from session state
                date_range = st.session_state.get("date_range", (date.today() - timedelta(days=365), date.today()))
                start_date, end_date = date_range

                # Train model
                classifier = GradientBoostClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
                comparison = classifier.train_and_compare(target_symbol, start_date, end_date)

                if comparison is None:
                    st.error(f"Insufficient data to train model for {target_symbol}")
                    return

                # Build feature dict
                features = {
                    "goldstein_mean": goldstein_mean,
                    "goldstein_min": goldstein_min,
                    "goldstein_max": goldstein_max,
                    "goldstein_std": abs(goldstein_max - goldstein_min) / 2,
                    "mentions_total": mentions_total,
                    "mentions_max": mentions_total // 3,
                    "avg_tone": avg_tone,
                    "conflict_count": conflict_count,
                    "cooperation_count": cooperation_count,
                }

                # Get predictions from both models
                pred_xgb = classifier.predict(target_symbol, features, "xgboost")
                pred_lgb = classifier.predict(target_symbol, features, "lightgbm")

                # Display results
                st.markdown("---")
                st.subheader("Prediction Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**XGBoost Prediction**")
                    if pred_xgb:
                        direction = "📈 UP" if pred_xgb.prediction == "UP" else "📉 DOWN"
                        st.metric("Direction", direction)
                        st.metric("Probability (UP)", f"{pred_xgb.probability:.1%}")
                        st.metric("Confidence", pred_xgb.confidence.upper())
                        st.caption(f"Model CV Accuracy: {comparison.xgboost_metrics.cv_accuracy:.1%}")

                with col2:
                    st.markdown("**LightGBM Prediction**")
                    if pred_lgb:
                        direction = "📈 UP" if pred_lgb.prediction == "UP" else "📉 DOWN"
                        st.metric("Direction", direction)
                        st.metric("Probability (UP)", f"{pred_lgb.probability:.1%}")
                        st.metric("Confidence", pred_lgb.confidence.upper())
                        st.caption(f"Model CV Accuracy: {comparison.lightgbm_metrics.cv_accuracy:.1%}")

                # Agreement indicator
                if pred_xgb and pred_lgb:
                    if pred_xgb.prediction == pred_lgb.prediction:
                        st.success(f"✅ Both models agree: **{pred_xgb.prediction}**")
                    else:
                        st.warning("⚠️ Models disagree - higher uncertainty")

                # Gauge visualization
                fig = go.Figure()

                if pred_xgb:
                    fig.add_trace(go.Indicator(
                        mode="gauge+number",
                        value=pred_xgb.probability * 100,
                        domain={"x": [0, 0.45], "y": [0, 1]},
                        title={"text": "XGBoost P(UP)"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "green" if pred_xgb.prediction == "UP" else "red"},
                            "threshold": {"line": {"color": "black", "width": 2}, "value": 50},
                        },
                    ))

                if pred_lgb:
                    fig.add_trace(go.Indicator(
                        mode="gauge+number",
                        value=pred_lgb.probability * 100,
                        domain={"x": [0.55, 1], "y": [0, 1]},
                        title={"text": "LightGBM P(UP)"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "green" if pred_lgb.prediction == "UP" else "red"},
                            "threshold": {"line": {"color": "black", "width": 2}, "value": 50},
                        },
                    ))

                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")


def render_training_interface():
    """Train and evaluate models."""
    st.subheader("Train & Evaluate Models")

    # Date range
    col1, col2 = st.columns(2)
    with col1:
        train_start = st.date_input(
            "Training Start Date",
            value=date.today() - timedelta(days=365),
        )
    with col2:
        train_end = st.date_input(
            "Training End Date",
            value=date.today(),
        )

    # Symbol selection
    all_symbols = get_all_symbols()
    selected_symbols = st.multiselect(
        "Symbols to Train",
        options=all_symbols,
        default=all_symbols[:5],
    )

    # Model selection
    col1, col2 = st.columns(2)
    with col1:
        run_gb = st.checkbox("Train Gradient Boosting", value=GB_AVAILABLE, disabled=not GB_AVAILABLE)
    with col2:
        run_lstm = st.checkbox("Train LSTM", value=False, disabled=not LSTM_AVAILABLE)

    if st.button("🚀 Train Models", type="primary", use_container_width=True):
        results = []

        # Gradient Boosting
        if run_gb and GB_AVAILABLE:
            st.markdown("### Gradient Boosting Results")
            progress = st.progress(0)

            classifier = GradientBoostClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)

            for i, symbol in enumerate(selected_symbols):
                with st.spinner(f"Training {symbol}..."):
                    try:
                        comparison = classifier.train_and_compare(symbol, train_start, train_end)
                        if comparison:
                            results.append({
                                "Symbol": symbol,
                                "Model": "XGBoost",
                                "CV Accuracy": comparison.xgboost_metrics.cv_accuracy,
                                "F1 Score": comparison.xgboost_metrics.f1_score,
                                "AUC": comparison.xgboost_metrics.roc_auc,
                            })
                            results.append({
                                "Symbol": symbol,
                                "Model": "LightGBM",
                                "CV Accuracy": comparison.lightgbm_metrics.cv_accuracy,
                                "F1 Score": comparison.lightgbm_metrics.f1_score,
                                "AUC": comparison.lightgbm_metrics.roc_auc,
                            })
                    except Exception as e:
                        st.warning(f"Failed for {symbol}: {e}")

                progress.progress((i + 1) / len(selected_symbols))

        # LSTM
        if run_lstm and LSTM_AVAILABLE:
            st.markdown("### LSTM Results")
            progress = st.progress(0)

            for i, symbol in enumerate(selected_symbols[:3]):  # Limit LSTM to 3 symbols (slow)
                with st.spinner(f"Training LSTM for {symbol}..."):
                    try:
                        dataset = MarketSequenceDataset(
                            symbol=symbol,
                            sequence_length=20,
                            start_date=train_start,
                            end_date=train_end,
                            test_ratio=0.2,
                        )

                        stats = dataset.get_stats()
                        if stats.train_size < 50:
                            st.warning(f"Insufficient data for {symbol}")
                            continue

                        model = MarketLSTM(
                            input_size=stats.n_features,
                            hidden_size=64,
                            num_layers=2,
                            dropout=0.2,
                        )

                        trainer = LSTMTrainer(model, learning_rate=0.001)
                        train_loader, test_loader = dataset.get_dataloaders(batch_size=32)

                        history = trainer.fit(
                            train_loader, test_loader,
                            epochs=30, early_stopping_patience=5, verbose=False
                        )

                        test_results = trainer.predict(test_loader)

                        results.append({
                            "Symbol": symbol,
                            "Model": "LSTM",
                            "CV Accuracy": test_results.accuracy,
                            "F1 Score": test_results.f1,
                            "AUC": test_results.auc,
                        })

                    except Exception as e:
                        st.warning(f"LSTM failed for {symbol}: {e}")

                progress.progress((i + 1) / min(len(selected_symbols), 3))

        # Display results
        if results:
            st.markdown("### Model Comparison")
            results_df = pd.DataFrame(results)

            # Summary table
            st.dataframe(
                results_df,
                column_config={
                    "CV Accuracy": st.column_config.NumberColumn("CV Accuracy", format="%.1%%"),
                    "F1 Score": st.column_config.NumberColumn("F1 Score", format="%.1%%"),
                    "AUC": st.column_config.NumberColumn("AUC", format="%.3f"),
                },
                hide_index=True,
                use_container_width=True,
            )

            # Chart
            fig = px.bar(
                results_df,
                x="Symbol",
                y="CV Accuracy",
                color="Model",
                barmode="group",
                title="Model Accuracy by Symbol",
            )
            fig.add_hline(y=0.5, line_dash="dash", annotation_text="Random Baseline (50%)")
            st.plotly_chart(fig, use_container_width=True)

            # Summary stats
            st.markdown("### Summary")
            col1, col2, col3 = st.columns(3)

            avg_acc = results_df["CV Accuracy"].mean()
            avg_auc = results_df["AUC"].mean()
            best = results_df.loc[results_df["CV Accuracy"].idxmax()]

            with col1:
                st.metric("Average Accuracy", f"{avg_acc:.1%}")
            with col2:
                st.metric("Average AUC", f"{avg_auc:.3f}")
            with col3:
                st.metric("Best Model", f"{best['Model']} on {best['Symbol']}")
        else:
            st.info("No results to display. Select models and symbols, then click Train.")


def render_feature_importance():
    """Show feature importance from trained models."""
    st.subheader("Feature Importance")

    if not GB_AVAILABLE:
        st.warning("Install `xgboost` and `lightgbm` to see feature importance.")
        return

    # Symbol selection
    all_symbols = get_all_symbols()
    symbol = st.selectbox("Select Symbol", options=all_symbols, index=0)

    if st.button("📊 Analyze Feature Importance", use_container_width=True):
        with st.spinner(f"Training model for {symbol}..."):
            try:
                date_range = st.session_state.get("date_range", (date.today() - timedelta(days=365), date.today()))
                start_date, end_date = date_range

                classifier = GradientBoostClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
                comparison = classifier.train_and_compare(symbol, start_date, end_date)

                if comparison is None:
                    st.error(f"Insufficient data for {symbol}")
                    return

                # XGBoost importance
                st.markdown("### XGBoost Feature Importance")
                xgb_importance = comparison.feature_importance_xgb

                xgb_df = pd.DataFrame([
                    {"Feature": k, "Importance": v}
                    for k, v in xgb_importance.items()
                ])

                fig1 = px.bar(
                    xgb_df.sort_values("Importance"),
                    x="Importance", y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="Blues",
                    title=f"XGBoost Feature Importance for {symbol}",
                )
                fig1.update_layout(height=400, coloraxis_showscale=False)
                st.plotly_chart(fig1, use_container_width=True)

                # LightGBM importance
                st.markdown("### LightGBM Feature Importance")
                lgb_importance = comparison.feature_importance_lgb

                lgb_df = pd.DataFrame([
                    {"Feature": k, "Importance": v}
                    for k, v in lgb_importance.items()
                ])

                fig2 = px.bar(
                    lgb_df.sort_values("Importance"),
                    x="Importance", y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="Greens",
                    title=f"LightGBM Feature Importance for {symbol}",
                )
                fig2.update_layout(height=400, coloraxis_showscale=False)
                st.plotly_chart(fig2, use_container_width=True)

                # Comparison
                st.markdown("### Feature Importance Comparison")
                comparison_df = pd.DataFrame({
                    "Feature": list(xgb_importance.keys()),
                    "XGBoost": list(xgb_importance.values()),
                    "LightGBM": [lgb_importance.get(k, 0) for k in xgb_importance.keys()],
                })

                st.dataframe(
                    comparison_df,
                    column_config={
                        "XGBoost": st.column_config.ProgressColumn("XGBoost", min_value=0, max_value=comparison_df["XGBoost"].max()),
                        "LightGBM": st.column_config.ProgressColumn("LightGBM", min_value=0, max_value=comparison_df["LightGBM"].max()),
                    },
                    hide_index=True,
                    use_container_width=True,
                )

                # Insights
                top_xgb = max(xgb_importance, key=xgb_importance.get)
                top_lgb = max(lgb_importance, key=lgb_importance.get)

                st.markdown(f"""
                ### Key Insights

                - **XGBoost's top feature:** `{top_xgb}` ({xgb_importance[top_xgb]:.3f})
                - **LightGBM's top feature:** `{top_lgb}` ({lgb_importance[top_lgb]:.3f})
                - **Model Agreement:** {"Same" if top_xgb == top_lgb else "Different"} top feature

                Higher importance means the feature has more influence on predictions.
                """)

            except Exception as e:
                st.error(f"Analysis failed: {e}")
