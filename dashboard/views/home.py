"""
Home Page - Dashboard Overview.

Shows key metrics and recent activity at a glance.
"""

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
from datetime import date, timedelta

from src.db.connection import get_session
from src.db.models import Event, MarketData, AnalysisResult
from src.config.constants import CAMEO_CATEGORIES, get_event_group
from sqlalchemy import func


def render():
    """Render the home page."""
    st.title("🌍 Geopolitical Market Tracker")
    st.markdown("Monitor geopolitical events and their impact on financial markets.")

    # Get date range from session state
    start_date, end_date = st.session_state.get(
        "date_range", (date.today() - timedelta(days=30), date.today())
    )

    # Key metrics row
    with get_session() as session:
        # Count events
        event_count = session.query(func.count(Event.id)).filter(
            Event.event_date >= start_date,
            Event.event_date <= end_date,
        ).scalar() or 0

        # Count market data points
        market_count = session.query(func.count(MarketData.id)).filter(
            MarketData.date >= start_date,
            MarketData.date <= end_date,
        ).scalar() or 0

        # Count significant analysis results
        significant_count = session.query(func.count(AnalysisResult.id)).filter(
            AnalysisResult.is_significant == True,
        ).scalar() or 0

        # Count anomalies
        anomaly_count = session.query(func.count(AnalysisResult.id)).filter(
            AnalysisResult.is_anomaly == True,
        ).scalar() or 0

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Geopolitical Events",
            value=f"{event_count:,}",
            help="Events detected in the selected date range",
        )

    with col2:
        st.metric(
            label="Market Data Points",
            value=f"{market_count:,}",
            help="OHLCV records for tracked symbols",
        )

    with col3:
        st.metric(
            label="Significant Results",
            value=f"{significant_count:,}",
            help="Event studies with p-value < 0.05",
        )

    with col4:
        st.metric(
            label="Anomalies Detected",
            value=f"{anomaly_count:,}",
            help="Unexplained moves or muted responses",
        )

    st.markdown("---")

    # Two-column layout for charts
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Events by Type")
        render_events_by_type_chart(start_date, end_date)

    with col_right:
        st.subheader("Events by Country")
        render_events_by_country_chart(start_date, end_date)

    st.markdown("---")

    # Recent significant events table
    st.subheader("Recent Significant Events")
    render_recent_events_table(start_date, end_date)


def render_events_by_type_chart(start_date: date, end_date: date):
    """Render bar chart of events by CAMEO category."""
    import plotly.express as px

    with get_session() as session:
        results = session.query(
            Event.event_root_code,
            func.count(Event.id).label("count"),
        ).filter(
            Event.event_date >= start_date,
            Event.event_date <= end_date,
        ).group_by(
            Event.event_root_code,
        ).all()

    if not results:
        st.info("No events found in this date range.")
        return

    # Convert to DataFrame and add category names
    df = pd.DataFrame(results, columns=["code", "count"])
    df["category"] = df["code"].apply(
        lambda x: CAMEO_CATEGORIES.get(str(x).zfill(2), "Unknown")
    )
    df["group"] = df["code"].apply(get_event_group)

    # Color by group
    color_map = {
        "verbal_cooperation": "#2ecc71",
        "material_cooperation": "#27ae60",
        "verbal_conflict": "#f39c12",
        "material_conflict": "#e74c3c",
        "violent_conflict": "#c0392b",
        "other": "#95a5a6",
    }

    fig = px.bar(
        df.sort_values("count", ascending=True),
        x="count",
        y="category",
        color="group",
        color_discrete_map=color_map,
        orientation="h",
        labels={"count": "Number of Events", "category": "Event Type"},
    )
    fig.update_layout(
        showlegend=True,
        legend_title="Event Group",
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_events_by_country_chart(start_date: date, end_date: date):
    """Render bar chart of events by country."""
    import plotly.express as px

    with get_session() as session:
        results = session.query(
            Event.action_geo_country_code,
            func.count(Event.id).label("count"),
        ).filter(
            Event.event_date >= start_date,
            Event.event_date <= end_date,
            Event.action_geo_country_code.isnot(None),
            Event.action_geo_country_code != "",
        ).group_by(
            Event.action_geo_country_code,
        ).order_by(
            func.count(Event.id).desc(),
        ).limit(15).all()

    if not results:
        st.info("No events with location data found.")
        return

    df = pd.DataFrame(results, columns=["country", "count"])

    fig = px.bar(
        df.sort_values("count", ascending=True),
        x="count",
        y="country",
        orientation="h",
        labels={"count": "Number of Events", "country": "Country Code"},
        color="count",
        color_continuous_scale="Reds",
    )
    fig.update_layout(
        showlegend=False,
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_showscale=False,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_recent_events_table(start_date: date, end_date: date, limit: int = 20):
    """Render table of recent significant events."""
    with get_session() as session:
        events = session.query(Event).filter(
            Event.event_date >= start_date,
            Event.event_date <= end_date,
            Event.num_mentions >= 10,  # Significant coverage
        ).order_by(
            Event.event_date.desc(),
            Event.num_mentions.desc(),
        ).limit(limit).all()

    if not events:
        st.info("No significant events found in this date range. Try ingesting more data.")
        return

    # Convert to DataFrame for display
    rows = []
    for e in events:
        rows.append({
            "Date": e.event_date,
            "Type": CAMEO_CATEGORIES.get(str(e.event_root_code).zfill(2), e.event_root_code),
            "Actor 1": e.actor1_name or e.actor1_code or "-",
            "Actor 2": e.actor2_name or e.actor2_code or "-",
            "Location": e.action_geo_name or e.action_geo_country_code or "-",
            "Goldstein": e.goldstein_scale or 0,
            "Mentions": e.num_mentions or 0,
        })

    df = pd.DataFrame(rows)

    # Style the Goldstein column
    st.dataframe(
        df,
        column_config={
            "Date": st.column_config.DateColumn("Date", width="small"),
            "Type": st.column_config.TextColumn("Event Type", width="medium"),
            "Actor 1": st.column_config.TextColumn("Actor 1", width="medium"),
            "Actor 2": st.column_config.TextColumn("Actor 2", width="medium"),
            "Location": st.column_config.TextColumn("Location", width="medium"),
            "Goldstein": st.column_config.NumberColumn(
                "Goldstein",
                help="Conflict (-10) to Cooperation (+10)",
                format="%.1f",
                width="small",
            ),
            "Mentions": st.column_config.NumberColumn(
                "Mentions",
                help="Number of media mentions",
                width="small",
            ),
        },
        hide_index=True,
        use_container_width=True,
    )
