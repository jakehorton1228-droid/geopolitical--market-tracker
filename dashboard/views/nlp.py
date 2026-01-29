"""
NLP Intelligence Page.

Provides natural language search, event classification, and
semantic analysis of geopolitical events.
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
from datetime import date, timedelta

from src.config.constants import CAMEO_CATEGORIES

# Check mode
USE_API = os.getenv("USE_API", "false").lower() == "true"

if not USE_API:
    from src.db.connection import get_session
    from src.db.models import Event


def render():
    """Render the NLP intelligence page."""
    st.title("NLP Intelligence")
    st.markdown(
        "Natural language search, event classification, and semantic analysis."
    )

    # Date range
    start_date, end_date = st.session_state.get(
        "date_range", (date.today() - timedelta(days=30), date.today())
    )

    tab1, tab2, tab3 = st.tabs(
        ["Semantic Search", "Event Classification", "Similar Events"]
    )

    with tab1:
        render_semantic_search(start_date, end_date)

    with tab2:
        render_classification(start_date, end_date)

    with tab3:
        render_similar_events(start_date, end_date)


def render_semantic_search(start_date: date, end_date: date):
    """Natural language search over events."""
    st.subheader("Semantic Search")
    st.markdown(
        "Search events using natural language queries. "
        "The system finds semantically similar events, not just keyword matches."
    )

    query = st.text_input(
        "Ask a question about events",
        placeholder="e.g., What military conflicts involved NATO this month?",
        key="rag_query",
    )

    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Max results", 5, 30, 10, key="rag_topk")
    with col2:
        min_relevance = st.slider(
            "Min relevance", 0.0, 0.8, 0.3, 0.05, key="rag_relevance"
        )

    if st.button("Search", type="primary", use_container_width=True) and query:
        with st.spinner("Building index and searching..."):
            try:
                from src.analysis.rag import EventRAG

                rag = EventRAG()
                n_indexed = rag.build_index(start_date, end_date, limit=500)

                if n_indexed == 0:
                    st.warning("No events found in the selected date range.")
                    return

                st.caption(f"Indexed {n_indexed} events")

                response = rag.query(query, top_k=top_k, min_relevance=min_relevance)

                if not response.sources:
                    st.info(f"No events matched your query with relevance >= {min_relevance:.0%}")
                    return

                # Display formatted response
                st.markdown("### Results")
                st.code(response.response)

                # Source events table
                st.markdown("### Source Events")
                rows = []
                for s in response.sources:
                    rows.append({
                        "Date": s.event_date,
                        "Event": s.text,
                        "Relevance": s.relevance_score,
                        "Goldstein": s.goldstein_scale,
                        "Mentions": s.num_mentions,
                        "Country": s.country,
                    })

                df = pd.DataFrame(rows)
                st.dataframe(
                    df,
                    column_config={
                        "Relevance": st.column_config.ProgressColumn(
                            "Relevance", min_value=0, max_value=1
                        ),
                        "Goldstein": st.column_config.NumberColumn(
                            "Goldstein", format="%.1f"
                        ),
                    },
                    hide_index=True,
                    use_container_width=True,
                )

            except ImportError as e:
                st.error(
                    f"Required packages not available: {e}. "
                    "Install: pip install sentence-transformers"
                )
            except Exception as e:
                st.error(f"Search failed: {e}")


def render_classification(start_date: date, end_date: date):
    """NLP-based event classification."""
    st.subheader("Event Classification")
    st.markdown(
        "Classify events into defense-relevant categories using "
        "zero-shot NLP classification."
    )

    col1, col2 = st.columns(2)
    with col1:
        max_events = st.slider("Events to classify", 10, 200, 50, key="nlp_max")
    with col2:
        use_transformer = st.checkbox(
            "Use transformer model (slower, more accurate)",
            value=False,
            key="nlp_use_transformer",
            help="If unchecked, uses keyword-based fallback (faster)",
        )

    if st.button("Classify Events", type="primary", use_container_width=True):
        with st.spinner("Classifying events..."):
            try:
                from src.analysis.nlp_classifier import EventClassifier

                classifier = EventClassifier()

                if not use_transformer:
                    # Force keyword fallback by not loading transformer
                    classifier._pipeline = "disabled"

                report = classifier.classify_date_range(
                    start_date, end_date, limit=max_events
                )

                if report.total_events == 0:
                    st.warning("No events found in the selected date range.")
                    return

                # Summary metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Events Classified", report.total_events)
                with col2:
                    st.metric("Avg Confidence", f"{report.avg_confidence:.0%}")

                # Category distribution
                st.markdown("### Category Distribution")
                if report.category_counts:
                    cat_df = pd.DataFrame(
                        [
                            {"Category": k, "Count": v}
                            for k, v in sorted(
                                report.category_counts.items(),
                                key=lambda x: x[1],
                                reverse=True,
                            )
                        ]
                    )

                    fig = px.bar(
                        cat_df,
                        x="Count",
                        y="Category",
                        orientation="h",
                        color="Count",
                        color_continuous_scale="Reds",
                        title="Events by Defense Category",
                    )
                    fig.update_layout(
                        height=max(300, len(cat_df) * 40),
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Detailed results table
                st.markdown("### Classification Details")
                rows = []
                for r in report.results:
                    rows.append({
                        "Date": r.event_date,
                        "Event": r.event_text[:80],
                        "Category": r.category,
                        "Confidence": r.confidence,
                    })

                results_df = pd.DataFrame(rows)
                st.dataframe(
                    results_df,
                    column_config={
                        "Confidence": st.column_config.ProgressColumn(
                            "Confidence", min_value=0, max_value=1
                        ),
                    },
                    hide_index=True,
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"Classification failed: {e}")


def render_similar_events(start_date: date, end_date: date):
    """Find similar events."""
    st.subheader("Find Similar Events")
    st.markdown(
        "Select an event and find historically similar events "
        "using semantic embedding similarity."
    )

    # Let user describe an event scenario
    st.markdown("**Describe an event scenario:**")

    col1, col2 = st.columns(2)
    with col1:
        actor1 = st.text_input("Actor 1", "Russia", key="sim_actor1")
        actor2 = st.text_input("Actor 2", "Ukraine", key="sim_actor2")
    with col2:
        root_code = st.selectbox(
            "Event Type",
            options=list(CAMEO_CATEGORIES.keys()),
            format_func=lambda x: f"{x}: {CAMEO_CATEGORIES[x]}",
            index=17,  # Default to "18: Use of force"
            key="sim_root_code",
        )
        goldstein = st.slider(
            "Goldstein Scale", -10.0, 10.0, -5.0, 0.5, key="sim_goldstein"
        )

    if st.button("Find Similar Events", type="primary", use_container_width=True):
        with st.spinner("Building index and searching..."):
            try:
                from src.analysis.rag import EventRAG

                rag = EventRAG()
                n_indexed = rag.build_index(start_date, end_date, limit=500)

                if n_indexed == 0:
                    st.warning("No events found to search.")
                    return

                # Build query event
                query_event = {
                    "id": -1,
                    "actor1_name": actor1,
                    "actor2_name": actor2,
                    "event_root_code": root_code,
                    "goldstein_scale": goldstein,
                    "avg_tone": goldstein / 2,
                }

                similar = rag.find_similar_events(query_event, top_k=10)

                if not similar:
                    st.info("No similar events found.")
                    return

                st.markdown(f"### Top {len(similar)} Similar Events")

                rows = []
                for s in similar:
                    rows.append({
                        "Date": s.event_date,
                        "Event": s.text,
                        "Similarity": s.relevance_score,
                        "Goldstein": s.goldstein_scale,
                        "Mentions": s.num_mentions,
                    })

                df = pd.DataFrame(rows)
                st.dataframe(
                    df,
                    column_config={
                        "Similarity": st.column_config.ProgressColumn(
                            "Similarity", min_value=0, max_value=1
                        ),
                        "Goldstein": st.column_config.NumberColumn(
                            "Goldstein", format="%.1f"
                        ),
                    },
                    hide_index=True,
                    use_container_width=True,
                )

            except ImportError:
                st.error(
                    "Required packages not available. "
                    "Install: pip install sentence-transformers"
                )
            except Exception as e:
                st.error(f"Search failed: {e}")
