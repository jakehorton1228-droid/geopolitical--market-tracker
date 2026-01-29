"""
Geopolitical Market Tracker Dashboard.

Main entry point for the Streamlit dashboard.

USAGE:
------
    # Local development (direct DB access)
    streamlit run dashboard/app.py

    # Containerized (via API)
    USE_API=true API_URL=http://api:8000 streamlit run dashboard/app.py

This dashboard provides:
1. Home - Overview of recent events and market moves
2. Event Map - Geographic visualization of geopolitical events
3. Market Analysis - Event study results and market reactions
4. Anomalies - Detected anomalies (unexplained moves, muted responses)
5. Predictions - Classification model predictions for market direction

ENVIRONMENT VARIABLES:
----------------------
    USE_API: Set to "true" to use API instead of direct DB (default: false)
    API_URL: API base URL when USE_API=true (default: http://localhost:8000)
"""

import os
import sys
from pathlib import Path

# Add project root to Python path so imports work correctly
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from datetime import date, timedelta

# Determine data access mode
USE_API = os.getenv("USE_API", "false").lower() == "true"

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Geopolitical Market Tracker",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import views (must be after set_page_config)
# Note: folder is named 'views' not 'pages' to prevent Streamlit auto-detection
from dashboard.views import home, event_map, market_analysis, anomalies, predictions, regression, explainability, nlp, monitoring


def main():
    """Main dashboard entry point."""

    # Sidebar navigation
    st.sidebar.title("🌍 Geopolitical Market Tracker")
    st.sidebar.markdown("---")

    # Navigation
    page = st.sidebar.radio(
        "Navigate",
        options=[
            "🏠 Home",
            "🗺️ Event Map",
            "📈 Market Analysis",
            "📊 Regression",
            "🔍 Anomalies",
            "🎯 Predictions",
            "🧠 Explainability",
            "🔬 NLP Intelligence",
            "📡 Monitoring",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")

    # Global date filter in sidebar
    st.sidebar.subheader("Date Range")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "From",
            value=date.today() - timedelta(days=30),
            max_value=date.today(),
            key="global_start_date",
        )
    with col2:
        end_date = st.date_input(
            "To",
            value=date.today(),
            max_value=date.today(),
            key="global_end_date",
        )

    # Store in session state for use by pages
    st.session_state["date_range"] = (start_date, end_date)

    st.sidebar.markdown("---")
    st.sidebar.caption("Built with Streamlit + Python")
    st.sidebar.caption("Data: GDELT + Yahoo Finance")

    # Show data access mode
    if USE_API:
        api_url = os.getenv("API_URL", "http://localhost:8000")
        st.sidebar.caption(f"Mode: API ({api_url})")
    else:
        st.sidebar.caption("Mode: Direct DB")

    # Route to selected page
    if page == "🏠 Home":
        home.render()
    elif page == "🗺️ Event Map":
        event_map.render()
    elif page == "📈 Market Analysis":
        market_analysis.render()
    elif page == "📊 Regression":
        regression.render()
    elif page == "🔍 Anomalies":
        anomalies.render()
    elif page == "🎯 Predictions":
        predictions.render()
    elif page == "🧠 Explainability":
        explainability.render()
    elif page == "🔬 NLP Intelligence":
        nlp.render()
    elif page == "📡 Monitoring":
        monitoring.render()


if __name__ == "__main__":
    main()
