"""
Anomalies Page - Detection Results.

Shows detected market anomalies:
- Unexplained moves: Large market moves without major events
- Muted responses: Big events with surprisingly small reactions
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

from src.db.connection import get_session
from src.db.models import Event, MarketData, AnalysisResult
from src.config.constants import CAMEO_CATEGORIES, get_symbol_info
from sqlalchemy import func


def render():
    """Render the anomalies page."""
    st.title("🔍 Anomaly Detection")
    st.markdown("Identify unusual market behavior that deviates from expected patterns.")

    # Explanation
    with st.expander("What are market anomalies?", expanded=False):
        st.markdown("""
        We detect two main types of anomalies:

        **1. Unexplained Moves** 📈
        - Market makes a large move (|return| > threshold)
        - No significant geopolitical event to explain it
        - May indicate: insider trading, technical factors, or unreported news

        **2. Muted Responses** 😐
        - Major geopolitical event occurs (high Goldstein score)
        - Expected market reaction is minimal
        - May indicate: market already priced in, event not relevant, or delayed reaction

        **Z-Score** measures how unusual a move is in standard deviations.
        Higher absolute Z-score = more unusual.
        """)

    # Get date range from session state
    start_date, end_date = st.session_state.get(
        "date_range", (date.today() - timedelta(days=30), date.today())
    )

    # Tabs for different anomaly types
    tab1, tab2, tab3 = st.tabs(["🔴 All Anomalies", "📈 Unexplained Moves", "😐 Muted Responses"])

    with tab1:
        render_all_anomalies()

    with tab2:
        render_unexplained_moves(start_date, end_date)

    with tab3:
        render_muted_responses()


def render_all_anomalies():
    """Show all detected anomalies from the database."""
    st.subheader("Detected Anomalies")

    with get_session() as session:
        anomalies = session.query(
            AnalysisResult, Event
        ).join(Event).filter(
            AnalysisResult.is_anomaly == True,
        ).order_by(
            AnalysisResult.anomaly_score.desc()
        ).limit(100).all()

    if not anomalies:
        st.info("No anomalies have been detected yet. Run the anomaly detection analysis to find them.")

        with st.expander("How to run anomaly detection"):
            st.code("""
from src.analysis.anomaly_detection import ProductionAnomalyDetector
from src.db.connection import get_session
from datetime import date

# Initialize detector
detector = ProductionAnomalyDetector()

# Detect anomalies for a date range
with get_session() as session:
    anomalies = detector.detect_anomalies(
        session,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
    )
            """, language="python")
        return

    rows = []
    for result, event in anomalies:
        rows.append({
            "Date": event.event_date,
            "Type": result.anomaly_type or "unknown",
            "Symbol": result.symbol,
            "Z-Score": result.anomaly_score or 0,
            "Expected": result.expected_return * 100 if result.expected_return else 0,
            "Actual": result.actual_return * 100 if result.actual_return else 0,
            "Event": CAMEO_CATEGORIES.get(str(event.event_root_code).zfill(2), event.event_root_code),
            "Goldstein": event.goldstein_scale or 0,
        })

    df = pd.DataFrame(rows)

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        type_filter = st.selectbox(
            "Anomaly Type",
            options=["All"] + sorted(df["Type"].unique().tolist()),
        )
    with col2:
        min_zscore = st.slider(
            "Minimum Z-Score",
            min_value=0.0,
            max_value=5.0,
            value=2.0,
            step=0.5,
        )

    # Apply filters
    if type_filter != "All":
        df = df[df["Type"] == type_filter]
    df = df[df["Z-Score"].abs() >= min_zscore]

    if df.empty:
        st.warning("No anomalies match the current filters.")
        return

    # Display
    st.dataframe(
        df,
        column_config={
            "Z-Score": st.column_config.NumberColumn("Z-Score", format="%.2f"),
            "Expected": st.column_config.NumberColumn("Expected (%)", format="%.2f"),
            "Actual": st.column_config.NumberColumn("Actual (%)", format="%.2f"),
            "Goldstein": st.column_config.NumberColumn("Goldstein", format="%.1f"),
        },
        hide_index=True,
        use_container_width=True,
    )

    # Summary chart
    st.markdown("---")
    st.subheader("Anomaly Distribution")

    fig = px.histogram(
        df,
        x="Z-Score",
        color="Type",
        nbins=20,
        labels={"Z-Score": "Anomaly Z-Score", "count": "Count"},
        title="Distribution of Anomaly Scores",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_unexplained_moves(start_date: date, end_date: date):
    """Show large market moves without obvious catalysts."""
    st.subheader("Unexplained Market Moves")
    st.markdown("Days with large returns but no significant geopolitical events.")

    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        return_threshold = st.slider(
            "Return Threshold (%)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.5,
            help="Minimum absolute return to flag as 'large move'",
        )
    with col2:
        min_goldstein = st.slider(
            "Major Event Threshold",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
            help="Goldstein score that would explain the move",
        )

    with get_session() as session:
        # Get days with large returns
        large_moves = session.query(MarketData).filter(
            MarketData.date >= start_date,
            MarketData.date <= end_date,
            func.abs(MarketData.daily_return) >= return_threshold / 100,
        ).order_by(
            func.abs(MarketData.daily_return).desc()
        ).limit(50).all()

    if not large_moves:
        st.info(f"No market moves exceeding {return_threshold}% found in this period.")
        return

    rows = []
    for move in large_moves:
        # Check for major events on that day
        with get_session() as session:
            major_event = session.query(Event).filter(
                Event.event_date == move.date,
                func.abs(Event.goldstein_scale) >= min_goldstein,
            ).first()

        info = get_symbol_info(move.symbol)
        rows.append({
            "Date": move.date,
            "Symbol": move.symbol,
            "Name": info["name"] if info else move.symbol,
            "Return (%)": move.daily_return * 100 if move.daily_return else 0,
            "Has Major Event": "Yes" if major_event else "No",
            "Potentially Unexplained": "⚠️" if not major_event else "",
        })

    df = pd.DataFrame(rows)

    # Filter to unexplained only
    show_unexplained = st.checkbox("Show only potentially unexplained", value=True)
    if show_unexplained:
        df = df[df["Has Major Event"] == "No"]

    if df.empty:
        st.success("All large moves have associated major events!")
        return

    st.dataframe(
        df,
        column_config={
            "Return (%)": st.column_config.NumberColumn("Return (%)", format="%.2f"),
        },
        hide_index=True,
        use_container_width=True,
    )


def render_muted_responses():
    """Show major events with surprisingly small market reactions."""
    st.subheader("Muted Market Responses")
    st.markdown("Major geopolitical events that didn't move markets as expected.")

    with get_session() as session:
        # Get muted response anomalies
        muted = session.query(
            AnalysisResult, Event
        ).join(Event).filter(
            AnalysisResult.is_anomaly == True,
            AnalysisResult.anomaly_type == "muted_response",
        ).order_by(
            func.abs(Event.goldstein_scale).desc()
        ).limit(50).all()

    if not muted:
        st.info("No muted response anomalies detected yet.")

        # Show how this would work
        st.markdown("---")
        st.markdown("**What would trigger a muted response alert?**")
        st.markdown("""
        - Event with |Goldstein| > 5 (major event)
        - Expected market move > 1%
        - Actual market move < 0.5%
        - This suggests the event was already priced in or markets don't consider it relevant
        """)
        return

    rows = []
    for result, event in muted:
        rows.append({
            "Date": event.event_date,
            "Event": CAMEO_CATEGORIES.get(str(event.event_root_code).zfill(2), event.event_root_code),
            "Actors": f"{event.actor1_code or '-'} → {event.actor2_code or '-'}",
            "Goldstein": event.goldstein_scale or 0,
            "Symbol": result.symbol,
            "Expected (%)": result.expected_return * 100 if result.expected_return else 0,
            "Actual (%)": result.actual_return * 100 if result.actual_return else 0,
            "Surprise": "Very Low" if abs(result.actual_return or 0) < 0.005 else "Low",
        })

    df = pd.DataFrame(rows)

    st.dataframe(
        df,
        column_config={
            "Goldstein": st.column_config.NumberColumn("Goldstein", format="%.1f"),
            "Expected (%)": st.column_config.NumberColumn("Expected (%)", format="%.2f"),
            "Actual (%)": st.column_config.NumberColumn("Actual (%)", format="%.2f"),
        },
        hide_index=True,
        use_container_width=True,
    )

    # Visualization
    if len(df) > 5:
        st.markdown("---")
        fig = px.scatter(
            df,
            x="Goldstein",
            y="Actual (%)",
            size=df["Expected (%)"].abs(),
            color="Symbol",
            hover_data=["Event", "Actors"],
            title="Event Severity vs Market Reaction",
            labels={"Goldstein": "Event Severity (Goldstein)", "Actual (%)": "Actual Market Move (%)"},
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
