"""
Explainability Page - SHAP Model Explanations.

Shows why models make specific predictions using SHAP values.
Critical for defense applications where decisions must be interpretable.
"""

import os
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

from src.config.constants import SYMBOLS, get_all_symbols, get_symbol_info

# Check mode
USE_API = os.getenv("USE_API", "false").lower() == "true"

# Check if SHAP is available
try:
    from src.analysis.explainability import ModelExplainer
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def render():
    """Render the explainability page."""
    st.title("Model Explainability (SHAP)")
    st.markdown(
        "Understand **why** models make specific predictions. "
        "Essential for accountable AI in mission-critical applications."
    )

    if not SHAP_AVAILABLE:
        st.error(
            "SHAP not available. Install with: `pip install shap`"
        )
        return

    with st.expander("What is SHAP?", expanded=False):
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)** decomposes each prediction
        into contributions from each feature.

        - **Positive SHAP value**: Feature pushes prediction toward UP
        - **Negative SHAP value**: Feature pushes prediction toward DOWN
        - **Magnitude**: How much influence the feature has

        Unlike basic feature importance (which is global), SHAP gives
        **per-prediction** explanations - you can see exactly why the
        model predicted UP or DOWN for a specific set of inputs.

        This is based on Shapley values from cooperative game theory,
        providing mathematically consistent attributions.
        """)

    # Date range
    start_date, end_date = st.session_state.get(
        "date_range", (date.today() - timedelta(days=30), date.today())
    )

    # Use a wider training window for explainability
    train_start = end_date - timedelta(days=365)

    # Tabs
    tab1, tab2 = st.tabs(
        ["Global Feature Importance", "Explain a Prediction"]
    )

    with tab1:
        render_global_importance(train_start, end_date)

    with tab2:
        render_prediction_explanation(train_start, end_date)


def render_global_importance(start_date: date, end_date: date):
    """Show global SHAP feature importance."""
    st.subheader("Global Feature Importance (SHAP)")

    all_symbols = get_all_symbols()
    symbol = st.selectbox(
        "Select Symbol", options=all_symbols, index=0, key="shap_global_symbol"
    )

    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox(
            "Model", options=["xgboost", "lightgbm"], key="shap_global_model"
        )

    if st.button(
        "Compute SHAP Values", type="primary", use_container_width=True
    ):
        with st.spinner(f"Training {model_type} and computing SHAP values for {symbol}..."):
            try:
                explainer = ModelExplainer()
                result = explainer.explain_model(
                    symbol, start_date, end_date, model_type
                )

                if result is None:
                    st.error(f"Insufficient data for {symbol}")
                    return

                # Model performance
                st.metric("Model CV Accuracy", f"{result.cv_accuracy:.1%}")

                # Feature importance bar chart
                st.markdown("### Mean |SHAP| Values")
                st.markdown(
                    "Higher values = more influence on predictions overall."
                )

                importance_df = pd.DataFrame(
                    [
                        {"Feature": k, "Mean |SHAP|": v}
                        for k, v in result.feature_importance.items()
                    ]
                )

                fig = px.bar(
                    importance_df.sort_values("Mean |SHAP|"),
                    x="Mean |SHAP|",
                    y="Feature",
                    orientation="h",
                    color="Mean |SHAP|",
                    color_continuous_scale="Reds",
                    title=f"SHAP Feature Importance for {symbol} ({model_type})",
                )
                fig.update_layout(
                    height=max(300, len(importance_df) * 40),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig, use_container_width=True)

                # SHAP value distribution (beeswarm-like via plotly)
                shap_df = explainer.get_shap_summary_plot_data(
                    symbol, start_date, end_date, model_type
                )
                if shap_df is not None and not shap_df.empty:
                    st.markdown("### SHAP Value Distribution")
                    st.markdown(
                        "Shows how each feature's SHAP values are distributed "
                        "across all training samples."
                    )

                    # Box plot of SHAP values per feature
                    melted = shap_df.melt(
                        var_name="Feature", value_name="SHAP Value"
                    )
                    fig2 = px.box(
                        melted,
                        x="SHAP Value",
                        y="Feature",
                        orientation="h",
                        title="Distribution of SHAP Values by Feature",
                    )
                    fig2.add_vline(
                        x=0, line_dash="dash", line_color="gray"
                    )
                    fig2.update_layout(
                        height=max(300, len(result.feature_names) * 40)
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                # Feature interactions
                if result.interactions:
                    st.markdown("### Feature Interactions")
                    st.markdown(
                        "Feature pairs that have the strongest combined effect."
                    )

                    top_interactions = list(result.interactions.items())[:5]
                    interaction_df = pd.DataFrame(
                        top_interactions,
                        columns=["Feature Pair", "Interaction Strength"],
                    )
                    st.dataframe(
                        interaction_df,
                        column_config={
                            "Interaction Strength": st.column_config.NumberColumn(
                                format="%.4f"
                            )
                        },
                        hide_index=True,
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"SHAP analysis failed: {e}")


def render_prediction_explanation(start_date: date, end_date: date):
    """Explain a single prediction."""
    st.subheader("Explain a Prediction")
    st.markdown(
        "Input event characteristics and see exactly which features "
        "drive the prediction and by how much."
    )

    all_symbols = get_all_symbols()

    col1, col2 = st.columns(2)

    with col1:
        symbol = st.selectbox(
            "Symbol", options=all_symbols, index=0, key="shap_pred_symbol"
        )
        model_type = st.selectbox(
            "Model", options=["xgboost", "lightgbm"], key="shap_pred_model"
        )
        goldstein_mean = st.slider(
            "Goldstein Mean", -10.0, 10.0, -2.0, 0.5,
            key="shap_goldstein_mean",
        )
        goldstein_min = st.slider(
            "Goldstein Min", -10.0, 10.0, -5.0, 0.5,
            key="shap_goldstein_min",
        )

    with col2:
        goldstein_max = st.slider(
            "Goldstein Max", -10.0, 10.0, 2.0, 0.5,
            key="shap_goldstein_max",
        )
        mentions_total = st.number_input(
            "Media Mentions", 0, 10000, 500, key="shap_mentions"
        )
        avg_tone = st.slider(
            "Average Tone", -10.0, 10.0, -1.5, 0.5,
            key="shap_tone",
        )
        conflict_count = st.number_input(
            "Conflict Events", 0, 100, 3, key="shap_conflict"
        )

    if st.button(
        "Explain Prediction", type="primary", use_container_width=True
    ):
        with st.spinner("Computing SHAP explanation..."):
            try:
                features = {
                    "goldstein_mean": goldstein_mean,
                    "goldstein_min": goldstein_min,
                    "goldstein_max": goldstein_max,
                    "goldstein_std": abs(goldstein_max - goldstein_min) / 2,
                    "mentions_total": mentions_total,
                    "mentions_max": max(1, mentions_total // 3),
                    "avg_tone": avg_tone,
                    "conflict_count": conflict_count,
                    "cooperation_count": max(0, 5 - conflict_count),
                }

                explainer = ModelExplainer()
                explanation = explainer.explain_prediction(
                    symbol, features, start_date, end_date, model_type
                )

                if explanation is None:
                    st.error(f"Could not generate explanation for {symbol}")
                    return

                # Prediction result
                direction_icon = "UP" if explanation.prediction == "UP" else "DOWN"
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prediction", direction_icon)
                with col2:
                    st.metric("Probability (UP)", f"{explanation.probability:.1%}")

                # Waterfall-style chart
                st.markdown("### Feature Contributions")
                st.markdown(
                    "Each bar shows how much a feature pushed the prediction "
                    "toward UP (positive) or DOWN (negative)."
                )

                contribs = explanation.feature_contributions
                contribs_df = pd.DataFrame(
                    [
                        {"Feature": k, "SHAP Value": v}
                        for k, v in sorted(
                            contribs.items(),
                            key=lambda x: abs(x[1]),
                            reverse=True,
                        )
                    ]
                )

                fig = px.bar(
                    contribs_df,
                    x="SHAP Value",
                    y="Feature",
                    orientation="h",
                    color="SHAP Value",
                    color_continuous_scale=["red", "gray", "green"],
                    color_continuous_midpoint=0,
                    title=f"SHAP Contributions for {symbol} Prediction",
                )
                fig.add_vline(x=0, line_dash="dash", line_color="gray")
                fig.update_layout(
                    height=max(300, len(contribs_df) * 40),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Narrative explanation
                st.markdown("### Explanation")
                st.code(explanation.narrative)

            except Exception as e:
                st.error(f"Explanation failed: {e}")
