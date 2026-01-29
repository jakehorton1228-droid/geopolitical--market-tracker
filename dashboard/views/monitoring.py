"""
Monitoring & Data Quality Page.

Provides visibility into data freshness, feature drift,
model health, and data quality issues. Essential for
production ML systems.
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

from src.config.constants import get_all_symbols

# Check mode
USE_API = os.getenv("USE_API", "false").lower() == "true"


def render():
    """Render the monitoring page."""
    st.title("Monitoring & Data Quality")
    st.markdown(
        "Data freshness, feature drift detection, and quality checks "
        "for production reliability."
    )

    with st.expander("Why monitoring matters", expanded=False):
        st.markdown("""
        **Production ML systems degrade silently.** Without monitoring:
        - Data pipelines can fail without anyone noticing
        - Feature distributions can shift (drift), making models unreliable
        - Data quality issues propagate through the entire analysis chain

        This page provides:
        - **Data Quality**: Completeness, freshness, and validity checks
        - **Drift Detection**: Statistical tests for distribution shifts
        - **System Health**: Overview of all data sources
        """)

    # Date range
    start_date, end_date = st.session_state.get(
        "date_range", (date.today() - timedelta(days=30), date.today())
    )

    tab1, tab2, tab3 = st.tabs(
        ["Data Quality", "Drift Detection", "System Health"]
    )

    with tab1:
        render_data_quality(start_date, end_date)

    with tab2:
        render_drift_detection(start_date, end_date)

    with tab3:
        render_system_health(start_date, end_date)


def render_data_quality(start_date: date, end_date: date):
    """Data quality scorecard and issue tracking."""
    st.subheader("Data Quality Scorecard")

    all_symbols = get_all_symbols()
    selected = st.multiselect(
        "Symbols to check",
        options=all_symbols,
        default=all_symbols[:5] if len(all_symbols) >= 5 else all_symbols,
        key="dq_symbols",
    )

    if not selected:
        st.info("Select at least one symbol.")
        return

    if st.button("Run Quality Checks", type="primary", use_container_width=True):
        with st.spinner("Checking data quality..."):
            try:
                from src.analysis.data_quality import DataQualityChecker

                checker = DataQualityChecker()
                report = checker.check_all(start_date, end_date, symbols=selected)

                # Overall score
                _display_quality_score(report.overall_score)

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Symbols Checked", report.symbols_checked)
                with col2:
                    st.metric("Total Issues", report.total_issues)
                with col3:
                    st.metric(
                        "High Severity",
                        report.high_severity_issues,
                        delta=None if report.high_severity_issues == 0
                        else f"-{report.high_severity_issues}",
                        delta_color="inverse",
                    )
                with col4:
                    st.metric("Symbols with Issues", report.symbols_with_issues)

                # Per-symbol quality table
                st.markdown("### Per-Symbol Quality")
                if report.symbol_reports:
                    rows = []
                    for r in report.symbol_reports:
                        rows.append({
                            "Symbol": r.symbol,
                            "Quality": r.quality_score,
                            "Completeness": r.completeness_score,
                            "Freshness": r.freshness_score,
                            "Validity": r.validity_score,
                            "Records": r.total_records,
                            "Missing Days": r.missing_days,
                            "Last Update": r.last_update,
                            "Days Stale": r.days_stale,
                            "Issues": len(r.issues),
                        })

                    df = pd.DataFrame(rows).sort_values("Quality")

                    st.dataframe(
                        df,
                        column_config={
                            "Quality": st.column_config.ProgressColumn(
                                "Quality", min_value=0, max_value=1
                            ),
                            "Completeness": st.column_config.ProgressColumn(
                                "Complete", min_value=0, max_value=1
                            ),
                            "Freshness": st.column_config.ProgressColumn(
                                "Fresh", min_value=0, max_value=1
                            ),
                            "Validity": st.column_config.ProgressColumn(
                                "Valid", min_value=0, max_value=1
                            ),
                        },
                        hide_index=True,
                        use_container_width=True,
                    )

                    # Quality bar chart
                    fig = px.bar(
                        df,
                        x="Quality",
                        y="Symbol",
                        orientation="h",
                        color="Quality",
                        color_continuous_scale=["#FF4B4B", "#FFA500", "#00CC00"],
                        title="Data Quality by Symbol",
                    )
                    fig.add_vline(x=0.8, line_dash="dash", line_color="green",
                                  annotation_text="Target (80%)")
                    fig.update_layout(
                        height=max(250, len(df) * 35),
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Issues table
                all_issues = []
                for r in report.symbol_reports:
                    for issue in r.issues:
                        all_issues.append({
                            "Severity": issue.severity.upper(),
                            "Type": issue.issue_type,
                            "Symbol": issue.affected_symbol or "N/A",
                            "Description": issue.description,
                            "Date": issue.affected_date,
                        })

                if all_issues:
                    st.markdown("### Issues Found")
                    issues_df = pd.DataFrame(all_issues)
                    # Sort by severity
                    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
                    issues_df["_sort"] = issues_df["Severity"].map(severity_order)
                    issues_df = issues_df.sort_values("_sort").drop(columns=["_sort"])

                    st.dataframe(
                        issues_df,
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.success("No data quality issues found.")

                # Full report
                with st.expander("Full Quality Report"):
                    st.code(report.summary)

            except Exception as e:
                st.error(f"Quality check failed: {e}")


def render_drift_detection(start_date: date, end_date: date):
    """Feature drift detection and visualization."""
    st.subheader("Feature Drift Detection")
    st.markdown(
        "Compares feature distributions between a baseline (historical) window "
        "and a recent test window. Significant shifts may indicate the model "
        "needs retraining."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        all_symbols = get_all_symbols()
        symbol = st.selectbox("Symbol", options=all_symbols, key="drift_symbol")
    with col2:
        baseline_days = st.slider(
            "Baseline window (days)", 30, 120, 60, key="drift_baseline"
        )
    with col3:
        test_days = st.slider(
            "Test window (days)", 7, 30, 14, key="drift_test"
        )

    if st.button("Detect Drift", type="primary", use_container_width=True):
        with st.spinner(f"Analyzing drift for {symbol}..."):
            try:
                from src.analysis.drift_detection import DriftDetector

                detector = DriftDetector(
                    baseline_window=baseline_days,
                    test_window=test_days,
                )
                report = detector.analyze(symbol, start_date, end_date)

                if report is None:
                    st.warning(f"Insufficient data for drift detection on {symbol}")
                    return

                # Overall status
                if report.drift_detected:
                    st.error(
                        f"Drift detected! Severity: **{report.overall_severity.upper()}** "
                        f"({report.drifted_features}/{report.total_features} features drifted)"
                    )
                else:
                    st.success(
                        f"No significant drift detected. "
                        f"All {report.total_features} features within expected ranges."
                    )

                # Feature drift details
                st.markdown("### Feature Drift Results")
                if report.feature_drifts:
                    rows = []
                    for f in report.feature_drifts:
                        rows.append({
                            "Feature": f.feature_name,
                            "Drift?": f.drift_detected,
                            "Severity": f.severity.upper(),
                            "PSI": f.psi_value,
                            "KS Statistic": f.ks_statistic,
                            "KS p-value": f.ks_pvalue,
                            "Baseline Mean": f.baseline_mean,
                            "Current Mean": f.current_mean,
                            "Mean Shift": f.current_mean - f.baseline_mean,
                        })

                    drift_df = pd.DataFrame(rows)

                    st.dataframe(
                        drift_df,
                        column_config={
                            "Drift?": st.column_config.CheckboxColumn("Drift?"),
                            "PSI": st.column_config.NumberColumn("PSI", format="%.4f"),
                            "KS Statistic": st.column_config.NumberColumn(
                                "KS Stat", format="%.4f"
                            ),
                            "KS p-value": st.column_config.NumberColumn(
                                "KS p-val", format="%.4f"
                            ),
                            "Baseline Mean": st.column_config.NumberColumn(
                                "Baseline", format="%.4f"
                            ),
                            "Current Mean": st.column_config.NumberColumn(
                                "Current", format="%.4f"
                            ),
                            "Mean Shift": st.column_config.NumberColumn(
                                "Shift", format="%.4f"
                            ),
                        },
                        hide_index=True,
                        use_container_width=True,
                    )

                    # PSI bar chart
                    fig = go.Figure()
                    colors = []
                    for _, row in drift_df.iterrows():
                        if row["Severity"] == "HIGH":
                            colors.append("#FF4B4B")
                        elif row["Severity"] == "MEDIUM":
                            colors.append("#FFA500")
                        elif row["Severity"] == "LOW":
                            colors.append("#FFD700")
                        else:
                            colors.append("#00CC00")

                    fig.add_trace(go.Bar(
                        x=drift_df["PSI"],
                        y=drift_df["Feature"],
                        orientation="h",
                        marker_color=colors,
                    ))
                    fig.add_vline(x=0.1, line_dash="dash", line_color="orange",
                                  annotation_text="Moderate (0.1)")
                    fig.add_vline(x=0.2, line_dash="dash", line_color="red",
                                  annotation_text="Significant (0.2)")
                    fig.update_layout(
                        title="Population Stability Index by Feature",
                        xaxis_title="PSI",
                        height=max(300, len(drift_df) * 35),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Mean shift chart
                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(
                        name="Baseline Mean",
                        x=drift_df["Feature"],
                        y=drift_df["Baseline Mean"],
                        marker_color="#2196F3",
                    ))
                    fig2.add_trace(go.Bar(
                        name="Current Mean",
                        x=drift_df["Feature"],
                        y=drift_df["Current Mean"],
                        marker_color="#FF9800",
                    ))
                    fig2.update_layout(
                        title="Feature Mean Comparison: Baseline vs Current",
                        barmode="group",
                        height=400,
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                # Full report
                with st.expander("Full Drift Report"):
                    st.code(report.summary)

            except Exception as e:
                st.error(f"Drift detection failed: {e}")

    # Multi-symbol comparison
    st.markdown("---")
    st.markdown("### Cross-Symbol Drift Comparison")

    all_symbols = get_all_symbols()
    compare_symbols = st.multiselect(
        "Compare symbols",
        options=all_symbols,
        default=all_symbols[:5] if len(all_symbols) >= 5 else all_symbols,
        key="drift_compare",
    )

    if compare_symbols and st.button("Compare Drift", use_container_width=True):
        with st.spinner("Comparing drift across symbols..."):
            try:
                from src.analysis.drift_detection import DriftDetector

                detector = DriftDetector(
                    baseline_window=baseline_days,
                    test_window=test_days,
                )
                comparison_df = detector.compare_symbols(
                    compare_symbols, start_date, end_date
                )

                if comparison_df.empty:
                    st.warning("No results. Ensure data is available.")
                    return

                st.dataframe(
                    comparison_df,
                    column_config={
                        "drift_detected": st.column_config.CheckboxColumn("Drift?"),
                        "drift_ratio": st.column_config.ProgressColumn(
                            "Drift Ratio", min_value=0, max_value=1
                        ),
                    },
                    hide_index=True,
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"Comparison failed: {e}")


def render_system_health(start_date: date, end_date: date):
    """Overall system health overview."""
    st.subheader("System Health Overview")

    if st.button("Check System Health", type="primary", use_container_width=True):
        with st.spinner("Checking system health..."):
            try:
                from src.analysis.data_quality import DataQualityChecker
                from src.db.connection import get_session
                from src.db.models import Event, MarketData

                checker = DataQualityChecker()

                # Quick staleness check for all symbols
                all_symbols = get_all_symbols()
                freshness_data = []

                for symbol in all_symbols:
                    _, days_stale, last_update, _ = checker.check_freshness(symbol)
                    freshness_data.append({
                        "Symbol": symbol,
                        "Last Update": last_update,
                        "Days Stale": days_stale,
                        "Status": (
                            "Fresh" if days_stale <= 3
                            else "Stale" if days_stale <= 7
                            else "Very Stale"
                        ),
                    })

                fresh_df = pd.DataFrame(freshness_data)

                # Summary counts
                fresh_count = sum(1 for d in freshness_data if d["Status"] == "Fresh")
                stale_count = sum(1 for d in freshness_data if d["Status"] == "Stale")
                very_stale_count = sum(1 for d in freshness_data if d["Status"] == "Very Stale")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fresh", fresh_count)
                with col2:
                    st.metric("Stale", stale_count)
                with col3:
                    st.metric("Very Stale", very_stale_count)

                # Database record counts
                st.markdown("### Database Records")
                with get_session() as session:
                    from sqlalchemy import func
                    event_count = session.query(func.count(Event.id)).scalar()
                    market_count = session.query(func.count(MarketData.id)).scalar()

                    # Date range
                    min_event_date = session.query(func.min(Event.event_date)).scalar()
                    max_event_date = session.query(func.max(Event.event_date)).scalar()
                    min_market_date = session.query(func.min(MarketData.date)).scalar()
                    max_market_date = session.query(func.max(MarketData.date)).scalar()

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Events", f"{event_count:,}")
                    if min_event_date and max_event_date:
                        st.caption(f"Range: {min_event_date} to {max_event_date}")
                with col2:
                    st.metric("Total Market Records", f"{market_count:,}")
                    if min_market_date and max_market_date:
                        st.caption(f"Range: {min_market_date} to {max_market_date}")

                # Freshness table
                st.markdown("### Data Freshness by Symbol")
                st.dataframe(
                    fresh_df,
                    hide_index=True,
                    use_container_width=True,
                )

                # Freshness visualization
                fig = px.bar(
                    fresh_df.sort_values("Days Stale", ascending=False),
                    x="Days Stale",
                    y="Symbol",
                    orientation="h",
                    color="Status",
                    color_discrete_map={
                        "Fresh": "#00CC00",
                        "Stale": "#FFA500",
                        "Very Stale": "#FF4B4B",
                    },
                    title="Data Staleness by Symbol",
                )
                fig.update_layout(
                    height=max(300, len(fresh_df) * 30),
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Health check failed: {e}")


def _display_quality_score(score: float):
    """Display a large quality score indicator."""
    if score >= 0.8:
        color = "#00CC00"
        label = "HEALTHY"
    elif score >= 0.5:
        color = "#FFA500"
        label = "DEGRADED"
    else:
        color = "#FF4B4B"
        label = "CRITICAL"

    st.markdown(
        f"""
        <div style="text-align: center; padding: 20px; border-radius: 10px;
                    background-color: {color}22; border: 2px solid {color};">
            <h1 style="color: {color}; margin: 0;">{score:.0%}</h1>
            <p style="color: {color}; margin: 0; font-size: 1.2em;">{label}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
