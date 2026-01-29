"""
Market Analysis Page - Event Study Results.

Shows how markets have reacted to geopolitical events.
"""

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta

from src.config.constants import SYMBOLS, get_all_symbols, get_symbol_info, CAMEO_CATEGORIES

# Check mode
USE_API = os.getenv("USE_API", "false").lower() == "true"

if USE_API:
    from dashboard.api_client import get_client
else:
    from src.db.connection import get_session
    from src.db.models import Event, MarketData, AnalysisResult
    from sqlalchemy import func


def render():
    """Render the market analysis page."""
    st.title("📈 Market Analysis")
    st.markdown("Analyze market reactions to geopolitical events using event study methodology.")

    # Get date range from session state
    start_date, end_date = st.session_state.get(
        "date_range", (date.today() - timedelta(days=30), date.today())
    )

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Market Overview", "🔬 Event Study Results", "📈 Price Charts"])

    with tab1:
        render_market_overview(start_date, end_date)

    with tab2:
        render_event_study_results(start_date, end_date)

    with tab3:
        render_price_charts(start_date, end_date)


def render_market_overview(start_date: date, end_date: date):
    """Show overview of market performance."""
    st.subheader("Market Performance Summary")

    with get_session() as session:
        # Get latest and first prices for each symbol to calculate period return
        symbols = get_all_symbols()

        returns_data = []
        for symbol in symbols:
            first_price = session.query(MarketData.close).filter(
                MarketData.symbol == symbol,
                MarketData.date >= start_date,
                MarketData.date <= end_date,
            ).order_by(MarketData.date.asc()).first()

            last_price = session.query(MarketData.close).filter(
                MarketData.symbol == symbol,
                MarketData.date >= start_date,
                MarketData.date <= end_date,
            ).order_by(MarketData.date.desc()).first()

            if first_price and last_price and first_price[0] and last_price[0]:
                period_return = (float(last_price[0]) - float(first_price[0])) / float(first_price[0]) * 100
                info = get_symbol_info(symbol)
                returns_data.append({
                    "symbol": symbol,
                    "name": info["name"] if info else symbol,
                    "category": info["category"] if info else "other",
                    "return": period_return,
                })

    if not returns_data:
        st.info("No market data found. Please run the market data ingestion first.")
        return

    df = pd.DataFrame(returns_data)

    # Display by category
    for category in df["category"].unique():
        st.markdown(f"**{category.replace('_', ' ').title()}**")

        cat_df = df[df["category"] == category].copy()

        # Color returns
        fig = px.bar(
            cat_df.sort_values("return"),
            x="return",
            y="name",
            orientation="h",
            color="return",
            color_continuous_scale=["red", "gray", "green"],
            color_continuous_midpoint=0,
            labels={"return": "Period Return (%)", "name": ""},
        )
        fig.update_layout(
            height=max(200, len(cat_df) * 35),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)


def render_event_study_results(start_date: date, end_date: date):
    """Show event study analysis results."""
    st.subheader("Event Study Results")

    st.markdown("""
    Event study methodology measures **Cumulative Abnormal Returns (CAR)** around geopolitical events.
    - Positive CAR = Market moved up more than expected
    - Negative CAR = Market moved down more than expected
    - Significant = p-value < 0.05
    """)

    with get_session() as session:
        # Get analysis results with event info
        results = session.query(
            AnalysisResult, Event
        ).join(Event).filter(
            Event.event_date >= start_date,
            Event.event_date <= end_date,
            AnalysisResult.analysis_type == "event_study",
        ).order_by(
            func.abs(AnalysisResult.car).desc()
        ).limit(100).all()

    if not results:
        st.info("No event study results found. Run the analysis module to generate results.")

        # Show how to run analysis
        with st.expander("How to run event study analysis"):
            st.code("""
from src.analysis.production_event_study import ProductionEventStudy
from src.db.connection import get_session
from datetime import date

# Initialize analyzer
analyzer = ProductionEventStudy()

# Analyze events for a date range
with get_session() as session:
    results = analyzer.analyze_date_range(
        session,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
    )
            """, language="python")
        return

    # Convert to DataFrame
    rows = []
    for result, event in results:
        rows.append({
            "Date": event.event_date,
            "Event Type": CAMEO_CATEGORIES.get(str(event.event_root_code).zfill(2), event.event_root_code),
            "Symbol": result.symbol,
            "CAR (%)": result.car * 100 if result.car else 0,
            "t-stat": result.car_t_stat or 0,
            "p-value": result.car_p_value or 1,
            "Significant": result.is_significant,
            "Actor 1": event.actor1_name or event.actor1_code or "-",
            "Actor 2": event.actor2_name or event.actor2_code or "-",
        })

    df = pd.DataFrame(rows)

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        show_significant_only = st.checkbox("Show significant only", value=False)
    with col2:
        symbol_filter = st.selectbox(
            "Filter by symbol",
            options=["All"] + sorted(df["Symbol"].unique().tolist()),
        )

    # Apply filters
    if show_significant_only:
        df = df[df["Significant"] == True]
    if symbol_filter != "All":
        df = df[df["Symbol"] == symbol_filter]

    if df.empty:
        st.warning("No results match the current filters.")
        return

    # Display results
    st.dataframe(
        df,
        column_config={
            "Date": st.column_config.DateColumn("Date"),
            "CAR (%)": st.column_config.NumberColumn("CAR (%)", format="%.2f"),
            "t-stat": st.column_config.NumberColumn("t-stat", format="%.2f"),
            "p-value": st.column_config.NumberColumn("p-value", format="%.4f"),
            "Significant": st.column_config.CheckboxColumn("Sig?"),
        },
        hide_index=True,
        use_container_width=True,
    )

    # Summary stats
    st.markdown("---")
    st.markdown("**Summary Statistics**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Results", len(df))
    with col2:
        st.metric("Significant", df["Significant"].sum())
    with col3:
        st.metric("Avg CAR", f"{df['CAR (%)'].mean():.2f}%")


def render_price_charts(start_date: date, end_date: date):
    """Show price charts for selected symbols."""
    st.subheader("Price Charts")

    # Symbol selector
    all_symbols = get_all_symbols()

    with get_session() as session:
        # Get symbols that have data
        available = session.query(MarketData.symbol).filter(
            MarketData.date >= start_date,
            MarketData.date <= end_date,
        ).distinct().all()
        available_symbols = [s[0] for s in available]

    symbols_with_data = [s for s in all_symbols if s in available_symbols]

    if not symbols_with_data:
        st.info("No market data available for the selected date range.")
        return

    selected_symbols = st.multiselect(
        "Select symbols to chart",
        options=symbols_with_data,
        default=symbols_with_data[:3] if len(symbols_with_data) >= 3 else symbols_with_data,
        max_selections=5,
    )

    if not selected_symbols:
        st.info("Select at least one symbol to display.")
        return

    # Fetch data
    with get_session() as session:
        data = session.query(MarketData).filter(
            MarketData.symbol.in_(selected_symbols),
            MarketData.date >= start_date,
            MarketData.date <= end_date,
        ).order_by(MarketData.date).all()

    if not data:
        st.warning("No data found for selected symbols.")
        return

    rows = []
    for d in data:
        info = get_symbol_info(d.symbol)
        rows.append({
            "date": d.date,
            "symbol": d.symbol,
            "name": info["name"] if info else d.symbol,
            "close": float(d.close) if d.close else None,
            "return": d.daily_return * 100 if d.daily_return else 0,
        })

    df = pd.DataFrame(rows)

    # Normalize prices to 100 at start
    normalized_df = df.copy()
    for symbol in selected_symbols:
        mask = normalized_df["symbol"] == symbol
        first_price = normalized_df.loc[mask, "close"].iloc[0] if mask.any() else 1
        if first_price and first_price > 0:
            normalized_df.loc[mask, "normalized"] = normalized_df.loc[mask, "close"] / first_price * 100

    # Price chart (normalized)
    fig = px.line(
        normalized_df,
        x="date",
        y="normalized",
        color="name",
        labels={"date": "Date", "normalized": "Indexed Price (100 = Start)", "name": "Symbol"},
        title="Normalized Price Comparison",
    )
    fig.update_layout(
        legend_title="Symbol",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Daily returns chart
    fig2 = px.bar(
        df,
        x="date",
        y="return",
        color="name",
        barmode="group",
        labels={"date": "Date", "return": "Daily Return (%)", "name": "Symbol"},
        title="Daily Returns",
    )
    fig2.update_layout(
        legend_title="Symbol",
        hovermode="x unified",
    )
    st.plotly_chart(fig2, use_container_width=True)
