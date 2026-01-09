"""
Predictions Page - Classification Model Results.

Shows market direction predictions based on geopolitical events.
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


def render():
    """Render the predictions page."""
    st.title("🎯 Market Direction Predictions")
    st.markdown("Predict whether markets will move UP or DOWN based on geopolitical events.")

    # Explanation
    with st.expander("How does the classifier work?", expanded=False):
        st.markdown("""
        **Classification Model Overview**

        We use Logistic Regression to predict market direction:
        - **Target**: Next-day return direction (UP = positive, DOWN = negative)
        - **Features**: Event characteristics (Goldstein score, event type, actors, etc.)

        **Training Process**:
        1. Collect historical events with associated market returns
        2. Extract features from each event
        3. Train separate models for each market symbol
        4. Cross-validate to measure accuracy

        **Interpretation**:
        - Probability > 0.5 → Predict UP
        - Probability < 0.5 → Predict DOWN
        - Confidence = how far from 0.5
        """)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Make Prediction", "📊 Model Performance", "📈 Feature Importance"])

    with tab1:
        render_prediction_interface()

    with tab2:
        render_model_performance()

    with tab3:
        render_feature_importance()


def render_prediction_interface():
    """Interactive interface to make predictions."""
    st.subheader("Predict Market Direction")
    st.markdown("Enter event characteristics to predict market impact.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Event Details**")

        event_type = st.selectbox(
            "Event Type",
            options=list(CAMEO_CATEGORIES.values()),
            index=15,  # Default to something in conflict range
        )

        goldstein = st.slider(
            "Goldstein Scale",
            min_value=-10.0,
            max_value=10.0,
            value=-3.0,
            step=0.5,
            help="-10 = extreme conflict, +10 = extreme cooperation",
        )

        num_mentions = st.number_input(
            "Number of Media Mentions",
            min_value=1,
            max_value=1000,
            value=50,
            help="Higher = more significant event",
        )

        avg_tone = st.slider(
            "Average Tone",
            min_value=-10.0,
            max_value=10.0,
            value=-2.0,
            step=0.5,
            help="Sentiment of media coverage",
        )

    with col2:
        st.markdown("**Target Market**")

        # Group symbols by category
        symbol_options = []
        for category, symbols in SYMBOLS.items():
            for symbol, name in symbols.items():
                symbol_options.append(f"{symbol} - {name}")

        selected = st.selectbox(
            "Select Market",
            options=symbol_options,
            index=0,
        )
        target_symbol = selected.split(" - ")[0]

        st.markdown("---")
        st.markdown("**Actor Information**")

        actor_country = st.text_input(
            "Primary Actor Country Code",
            value="USA",
            max_chars=3,
            help="3-letter ISO code",
        ).upper()

        is_conflict = st.checkbox(
            "Involves Conflict (CAMEO 14-20)",
            value=True,
        )

    # Make prediction button
    st.markdown("---")
    if st.button("🔮 Predict Market Direction", type="primary", use_container_width=True):
        # Build feature vector
        features = {
            "goldstein_scale": goldstein,
            "num_mentions": num_mentions,
            "avg_tone": avg_tone,
            "is_conflict": 1 if is_conflict else 0,
            "actor_country": actor_country,
        }

        # Map event type back to code for group
        event_code = None
        for code, name in CAMEO_CATEGORIES.items():
            if name == event_type:
                event_code = code
                break

        if event_code:
            event_group = get_event_group(event_code)
            features["event_group"] = event_group

        # Simulate prediction (in production, would load trained model)
        prediction, probability = simulate_prediction(features, target_symbol)

        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            direction = "📈 UP" if prediction == 1 else "📉 DOWN"
            st.metric("Predicted Direction", direction)

        with col2:
            confidence = abs(probability - 0.5) * 2 * 100
            st.metric("Confidence", f"{confidence:.1f}%")

        with col3:
            prob_display = probability * 100
            st.metric("P(UP)", f"{prob_display:.1f}%")

        # Visualization
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": "green" if prediction == 1 else "red"},
                "steps": [
                    {"range": [0, 30], "color": "lightcoral"},
                    {"range": [30, 50], "color": "lightyellow"},
                    {"range": [50, 70], "color": "lightyellow"},
                    {"range": [70, 100], "color": "lightgreen"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 2},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
            title={"text": f"P(UP) for {target_symbol}"},
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Disclaimer
        st.caption("⚠️ This is a simplified demonstration. Real predictions require a trained model and more features.")


def simulate_prediction(features: dict, symbol: str) -> tuple[int, float]:
    """
    Simulate a prediction (placeholder for actual model).

    In production, this would load a trained model and make real predictions.
    For now, we use simple heuristics based on the features.
    """
    # Simple heuristic-based simulation
    base_prob = 0.5

    # Goldstein effect: negative = down, positive = up
    goldstein_effect = features.get("goldstein_scale", 0) * 0.03
    base_prob += goldstein_effect

    # Tone effect
    tone_effect = features.get("avg_tone", 0) * 0.02
    base_prob += tone_effect

    # Conflict tends to push down
    if features.get("is_conflict", 0):
        base_prob -= 0.1

    # More mentions = stronger effect
    mentions = features.get("num_mentions", 0)
    if mentions > 100:
        base_prob += (goldstein_effect * 0.5)  # Amplify Goldstein effect

    # Clamp probability
    probability = max(0.1, min(0.9, base_prob))

    # Add some randomness for realism
    probability += np.random.normal(0, 0.05)
    probability = max(0.1, min(0.9, probability))

    prediction = 1 if probability > 0.5 else 0

    return prediction, probability


def render_model_performance():
    """Show model performance metrics."""
    st.subheader("Model Performance Metrics")

    st.info("Model performance metrics will be available after training the classifier on historical data.")

    # Show what metrics would look like
    st.markdown("**Expected Metrics (after training)**")

    # Simulated metrics for demonstration
    metrics_df = pd.DataFrame([
        {"Symbol": "CL=F", "Accuracy": 0.54, "Precision": 0.55, "Recall": 0.52, "F1": 0.53, "AUC": 0.56},
        {"Symbol": "GC=F", "Accuracy": 0.52, "Precision": 0.53, "Recall": 0.51, "F1": 0.52, "AUC": 0.54},
        {"Symbol": "SPY", "Accuracy": 0.51, "Precision": 0.52, "Recall": 0.50, "F1": 0.51, "AUC": 0.53},
        {"Symbol": "^VIX", "Accuracy": 0.55, "Precision": 0.56, "Recall": 0.54, "F1": 0.55, "AUC": 0.58},
        {"Symbol": "EURUSD=X", "Accuracy": 0.50, "Precision": 0.51, "Recall": 0.49, "F1": 0.50, "AUC": 0.52},
    ])

    st.dataframe(
        metrics_df,
        column_config={
            "Accuracy": st.column_config.NumberColumn("Accuracy", format="%.2f"),
            "Precision": st.column_config.NumberColumn("Precision", format="%.2f"),
            "Recall": st.column_config.NumberColumn("Recall", format="%.2f"),
            "F1": st.column_config.NumberColumn("F1 Score", format="%.2f"),
            "AUC": st.column_config.NumberColumn("AUC-ROC", format="%.2f"),
        },
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("""
    **Interpreting the metrics:**
    - **Accuracy**: % of correct predictions (50% = random)
    - **Precision**: Of UP predictions, what % were actually UP
    - **Recall**: Of actual UP days, what % did we catch
    - **F1 Score**: Harmonic mean of Precision and Recall
    - **AUC-ROC**: Area under ROC curve (0.5 = random, 1.0 = perfect)

    ⚠️ Predicting market direction is extremely hard. Even 52-55% accuracy
    can be valuable if the predictions are calibrated and consistent.
    """)

    # How to train
    with st.expander("How to train the classifier"):
        st.code("""
from src.analysis.classification import ProductionClassifier
from src.db.connection import get_session
from datetime import date

# Initialize classifier
classifier = ProductionClassifier()

# Prepare training data
with get_session() as session:
    X, y = classifier.prepare_training_data(
        session,
        start_date=date(2023, 1, 1),
        end_date=date(2024, 12, 31),
        symbol="CL=F",
    )

# Train with cross-validation
results = classifier.train_with_cv(X, y, cv_folds=5)
print(f"Cross-validated accuracy: {results['mean_accuracy']:.2%}")

# Save model
classifier.save_model("models/classifier_clf.pkl")
        """, language="python")


def render_feature_importance():
    """Show which features matter most for predictions."""
    st.subheader("Feature Importance")

    st.info("Feature importance analysis will be available after training the classifier.")

    # Simulated feature importance for demonstration
    st.markdown("**Expected Feature Importance (after training)**")

    features_df = pd.DataFrame([
        {"Feature": "goldstein_scale", "Importance": 0.25, "Description": "Event conflict/cooperation score"},
        {"Feature": "num_mentions", "Importance": 0.18, "Description": "Media coverage volume"},
        {"Feature": "event_group_violent_conflict", "Importance": 0.15, "Description": "Is violent conflict event"},
        {"Feature": "avg_tone", "Importance": 0.12, "Description": "Media sentiment"},
        {"Feature": "actor1_is_major_power", "Importance": 0.10, "Description": "Involves USA/CHN/RUS"},
        {"Feature": "is_weekend_adjacent", "Importance": 0.08, "Description": "Friday or Monday event"},
        {"Feature": "num_sources", "Importance": 0.07, "Description": "Number of news sources"},
        {"Feature": "event_group_material_conflict", "Importance": 0.05, "Description": "Is material conflict"},
    ])

    fig = px.bar(
        features_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
        labels={"Importance": "Relative Importance", "Feature": ""},
        title="Feature Importance for Market Direction Prediction",
    )
    fig.update_layout(
        height=400,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        features_df[["Feature", "Description", "Importance"]],
        column_config={
            "Importance": st.column_config.ProgressColumn(
                "Importance",
                min_value=0,
                max_value=0.3,
                format="%.2f",
            ),
        },
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("""
    **Key Insights:**
    1. **Goldstein scale** is the most predictive feature - makes sense as it directly measures event severity
    2. **Media coverage** (mentions) amplifies the signal
    3. **Violent conflict** events have clear directional impact
    4. **Major power involvement** (USA, China, Russia) matters more than smaller nations
    5. **Weekend adjacency** may capture information gaps

    These insights help us understand what drives market reactions to geopolitical events.
    """)
