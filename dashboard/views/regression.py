"""
Regression Analysis Page.

Shows OLS regression results analyzing the relationship between
geopolitical event features and market returns.
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

from src.config.constants import get_all_symbols, get_symbol_info

# Check mode
USE_API = os.getenv("USE_API", "false").lower() == "true"

if USE_API:
    from dashboard.api_client import get_client


def render():
    """Render the regression analysis page."""
    st.title("Regression Analysis")
    st.markdown(
        "OLS regression measuring how geopolitical event features "
        "relate to market returns."
    )

    with st.expander("What does this show?", expanded=False):
        st.markdown("""
        **Ordinary Least Squares (OLS) Regression** fits a linear model:

        `return = b0 + b1*goldstein + b2*mentions + b3*tone + b4*conflict`

        **Key metrics:**
        - **R-squared**: How much variance in returns is explained (0-1)
        - **p-value**: Whether a predictor is statistically significant (< 0.05)
        - **Coefficient**: The direction and magnitude of the effect
        - **F-statistic**: Whether the overall model is significant

        Low R-squared is expected for daily returns - markets are affected by
        many factors beyond geopolitical events.
        """)

    # Date range
    start_date, end_date = st.session_state.get(
        "date_range", (date.today() - timedelta(days=30), date.today())
    )

    # Tabs
    tab1, tab2 = st.tabs(["Single Symbol", "Compare Markets"])

    with tab1:
        render_single_regression(start_date, end_date)

    with tab2:
        render_market_comparison(start_date, end_date)


def render_single_regression(start_date: date, end_date: date):
    """Run and display regression for a single symbol."""
    st.subheader("Regression Analysis")

    all_symbols = get_all_symbols()
    symbol = st.selectbox("Select Symbol", options=all_symbols, index=0)

    if st.button("Run Regression", type="primary", use_container_width=True):
        with st.spinner(f"Running regression for {symbol}..."):
            try:
                if USE_API:
                    client = get_client()
                    result = client._get(
                        f"/api/analysis/regression/{symbol}",
                        params={
                            "start_date": start_date.isoformat(),
                            "end_date": end_date.isoformat(),
                        },
                    )
                    if not result:
                        st.error(f"No regression results for {symbol}")
                        return
                else:
                    from src.analysis.production_regression import (
                        ProductionRegression,
                    )

                    regression = ProductionRegression()
                    raw = regression.analyze(symbol, start_date, end_date)

                    if raw is None:
                        st.error(f"Insufficient data for {symbol}")
                        return

                    result = {
                        "symbol": raw.symbol,
                        "r_squared": raw.r_squared,
                        "adj_r_squared": raw.adj_r_squared,
                        "f_statistic": raw.f_statistic,
                        "f_pvalue": raw.f_pvalue,
                        "coefficients": raw.coefficients,
                        "p_values": raw.p_values,
                        "conf_int_lower": raw.conf_int_lower,
                        "conf_int_upper": raw.conf_int_upper,
                        "n_observations": raw.n_observations,
                        "summary": raw.summary,
                    }

                _display_regression_result(result)

            except Exception as e:
                st.error(f"Regression failed: {e}")


def render_market_comparison(start_date: date, end_date: date):
    """Compare regression results across multiple symbols."""
    st.subheader("Market Comparison")

    all_symbols = get_all_symbols()
    selected = st.multiselect(
        "Select Symbols",
        options=all_symbols,
        default=all_symbols[:5] if len(all_symbols) >= 5 else all_symbols,
        max_selections=10,
    )

    if not selected:
        st.info("Select at least one symbol.")
        return

    if st.button("Compare All", type="primary", use_container_width=True):
        rows = []
        progress = st.progress(0)

        for i, symbol in enumerate(selected):
            try:
                if USE_API:
                    client = get_client()
                    result = client._get(
                        f"/api/analysis/regression/{symbol}",
                        params={
                            "start_date": start_date.isoformat(),
                            "end_date": end_date.isoformat(),
                        },
                    )
                else:
                    from src.analysis.production_regression import (
                        ProductionRegression,
                    )

                    regression = ProductionRegression()
                    raw = regression.analyze(symbol, start_date, end_date)
                    if raw is None:
                        continue
                    result = {
                        "r_squared": raw.r_squared,
                        "adj_r_squared": raw.adj_r_squared,
                        "f_pvalue": raw.f_pvalue,
                        "n_observations": raw.n_observations,
                        "p_values": raw.p_values,
                        "coefficients": raw.coefficients,
                    }

                if not result:
                    continue

                p_values = result.get("p_values", {})
                sig_predictors = [
                    name
                    for name, p in p_values.items()
                    if p < 0.05 and name != "const"
                ]

                info = get_symbol_info(symbol)
                rows.append(
                    {
                        "Symbol": symbol,
                        "Name": info["name"] if info else symbol,
                        "R-squared": result["r_squared"],
                        "Adj R-squared": result["adj_r_squared"],
                        "F p-value": result["f_pvalue"],
                        "Observations": result["n_observations"],
                        "Significant Predictors": ", ".join(sig_predictors)
                        or "None",
                        "Goldstein Coef": result.get("coefficients", {}).get(
                            "goldstein_mean", 0
                        ),
                        "Goldstein p-val": p_values.get("goldstein_mean", 1),
                    }
                )

            except Exception:
                pass

            progress.progress((i + 1) / len(selected))

        if not rows:
            st.warning("No results. Ensure data is ingested for the selected date range.")
            return

        df = pd.DataFrame(rows).sort_values("R-squared", ascending=False)

        st.dataframe(
            df,
            column_config={
                "R-squared": st.column_config.NumberColumn(
                    "R-squared", format="%.4f"
                ),
                "Adj R-squared": st.column_config.NumberColumn(
                    "Adj R-sq", format="%.4f"
                ),
                "F p-value": st.column_config.NumberColumn(
                    "F p-value", format="%.4f"
                ),
                "Goldstein Coef": st.column_config.NumberColumn(
                    "Goldstein Coef", format="%.6f"
                ),
                "Goldstein p-val": st.column_config.NumberColumn(
                    "Goldstein p-val", format="%.4f"
                ),
            },
            hide_index=True,
            use_container_width=True,
        )

        # R-squared chart
        fig = px.bar(
            df,
            x="R-squared",
            y="Name",
            orientation="h",
            color="R-squared",
            color_continuous_scale="Blues",
            title="R-squared by Symbol (higher = more variance explained)",
        )
        fig.update_layout(
            height=max(250, len(df) * 35),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)


def _display_regression_result(result: dict):
    """Display a single regression result."""
    # Model fit metrics
    st.markdown("### Model Fit")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("R-squared", f"{result['r_squared']:.4f}")
    with col2:
        st.metric("Adj R-squared", f"{result['adj_r_squared']:.4f}")
    with col3:
        st.metric("F-statistic", f"{result['f_statistic']:.2f}")
    with col4:
        f_sig = "Yes" if result["f_pvalue"] < 0.05 else "No"
        st.metric("Model Significant?", f_sig)

    # Coefficients table
    st.markdown("### Coefficients")
    coefficients = result.get("coefficients", {})
    p_values = result.get("p_values", {})
    ci_lower = result.get("conf_int_lower", {})
    ci_upper = result.get("conf_int_upper", {})

    coef_rows = []
    for name in coefficients:
        coef_rows.append(
            {
                "Variable": name,
                "Coefficient": coefficients[name],
                "p-value": p_values.get(name, 1),
                "CI Lower": ci_lower.get(name, 0),
                "CI Upper": ci_upper.get(name, 0),
                "Significant": p_values.get(name, 1) < 0.05,
            }
        )

    coef_df = pd.DataFrame(coef_rows)

    st.dataframe(
        coef_df,
        column_config={
            "Coefficient": st.column_config.NumberColumn(
                "Coefficient", format="%.6f"
            ),
            "p-value": st.column_config.NumberColumn("p-value", format="%.4f"),
            "CI Lower": st.column_config.NumberColumn("CI Lower", format="%.6f"),
            "CI Upper": st.column_config.NumberColumn("CI Upper", format="%.6f"),
            "Significant": st.column_config.CheckboxColumn("Sig?"),
        },
        hide_index=True,
        use_container_width=True,
    )

    # Coefficient bar chart (exclude constant)
    plot_df = coef_df[coef_df["Variable"] != "const"]
    if not plot_df.empty:
        fig = px.bar(
            plot_df,
            x="Coefficient",
            y="Variable",
            orientation="h",
            color="Significant",
            color_discrete_map={True: "#2196F3", False: "#CCCCCC"},
            title="Coefficient Values (blue = statistically significant)",
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Interpretation
    st.markdown("### Interpretation")
    sig_vars = [
        name
        for name, p in p_values.items()
        if p < 0.05 and name != "const"
    ]

    if sig_vars:
        for name in sig_vars:
            coef = coefficients[name]
            direction = "increases" if coef > 0 else "decreases"
            st.markdown(
                f"- **{name}**: When it increases by 1 unit, "
                f"the return {direction} by {abs(coef) * 100:.4f}%"
            )
    else:
        st.markdown(
            "No statistically significant predictors found. "
            "This is common for daily returns - geopolitical events "
            "explain only a small portion of market movements."
        )

    # Full statsmodels summary
    if "summary" in result:
        with st.expander("Full Statsmodels Output"):
            st.code(result["summary"])

    st.caption(f"Based on {result.get('n_observations', 'N/A')} observations")
