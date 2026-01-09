"""
Home Page - Dashboard Overview.

Shows key metrics and recent activity at a glance.
Supports both direct DB access and API mode.
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
from datetime import date, timedelta

from src.config.constants import CAMEO_CATEGORIES, get_event_group

# Check mode
USE_API = os.getenv("USE_API", "false").lower() == "true"

if USE_API:
    from dashboard.api_client import get_client
else:
    from src.db.connection import get_session
    from src.db.models import Event, MarketData, AnalysisResult
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
    if USE_API:
        client = get_client()
        event_count = client.get_events_count(start_date, end_date)
        # API doesn't have market count endpoint, estimate from symbols
        symbols_info = client.get_symbols()
        market_count = symbols_info.get("total", 0) * 30  # Rough estimate
        summary = client.get_analysis_summary()
        significant_count = summary.get("significant_results", 0)
        anomaly_count = summary.get("total_anomalies", 0)
    else:
        with get_session() as session:
            event_count = session.query(func.count(Event.id)).filter(
                Event.event_date >= start_date,
                Event.event_date <= end_date,
            ).scalar() or 0

            market_count = session.query(func.count(MarketData.id)).filter(
                MarketData.date >= start_date,
                MarketData.date <= end_date,
            ).scalar() or 0

            significant_count = session.query(func.count(AnalysisResult.id)).filter(
                AnalysisResult.is_significant == True,
            ).scalar() or 0

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

    if USE_API:
        client = get_client()
        results = client.get_events_by_type(start_date, end_date)
        if not results:
            st.info("No events found in this date range.")
            return
        df = pd.DataFrame(results)
        df = df.rename(columns={"code": "code", "name": "category", "group": "group", "count": "count"})
    else:
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

    if USE_API:
        client = get_client()
        results = client.get_events_by_country(start_date, end_date, limit=15)
        if not results:
            st.info("No events with location data found.")
            return
        df = pd.DataFrame(results)
        df = df.rename(columns={"country_code": "country", "count": "count"})
    else:
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
    if USE_API:
        client = get_client()
        events = client.get_events(
            start_date=start_date,
            end_date=end_date,
            min_mentions=10,
            limit=limit,
        )
        if not events:
            st.info("No significant events found in this date range. Try ingesting more data.")
            return

        rows = []
        for e in events:
            rows.append({
                "Date": e.get("event_date"),
                "Type": CAMEO_CATEGORIES.get(str(e.get("event_root_code", "")).zfill(2), e.get("event_root_code")),
                "Actor 1": e.get("actor1_name") or e.get("actor1_code") or "-",
                "Actor 2": e.get("actor2_name") or e.get("actor2_code") or "-",
                "Location": e.get("action_geo_name") or e.get("action_geo_country_code") or "-",
                "Goldstein": e.get("goldstein_scale") or 0,
                "Mentions": e.get("num_mentions") or 0,
            })
    else:
        with get_session() as session:
            events = session.query(Event).filter(
                Event.event_date >= start_date,
                Event.event_date <= end_date,
                Event.num_mentions >= 10,
            ).order_by(
                Event.event_date.desc(),
                Event.num_mentions.desc(),
            ).limit(limit).all()

        if not events:
            st.info("No significant events found in this date range. Try ingesting more data.")
            return

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
