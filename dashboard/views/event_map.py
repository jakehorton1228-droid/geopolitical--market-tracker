"""
Event Map Page - Geographic Visualization.

Shows events on an interactive world map using Folium.
"""

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from datetime import date, timedelta

from src.db.connection import get_session
from src.db.models import Event
from src.config.constants import CAMEO_CATEGORIES, get_event_group
from sqlalchemy import func


def render():
    """Render the event map page."""
    st.title("🗺️ Event Map")
    st.markdown("Geographic visualization of geopolitical events.")

    # Get date range from session state
    start_date, end_date = st.session_state.get(
        "date_range", (date.today() - timedelta(days=30), date.today())
    )

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        min_mentions = st.slider(
            "Minimum Mentions",
            min_value=5,
            max_value=100,
            value=10,
            help="Filter events by minimum media coverage",
        )

    with col2:
        event_groups = st.multiselect(
            "Event Groups",
            options=["verbal_cooperation", "material_cooperation", "verbal_conflict", "material_conflict", "violent_conflict"],
            default=["material_conflict", "violent_conflict"],
            help="Filter by event group",
        )

    with col3:
        country_filter = st.text_input(
            "Country Code (optional)",
            value="",
            max_chars=3,
            help="3-letter ISO code (e.g., USA, RUS, CHN)",
        ).upper()

    # Fetch events with location data
    events_df = fetch_events_with_location(
        start_date,
        end_date,
        min_mentions=min_mentions,
        event_groups=event_groups if event_groups else None,
        country_code=country_filter if country_filter else None,
    )

    if events_df.empty:
        st.warning("No events with geographic coordinates found for the selected filters.")
        st.info("Try adjusting the date range or filters, or ingest more GDELT data.")
        return

    st.markdown(f"**{len(events_df):,}** events with location data")

    # Render the map
    render_folium_map(events_df)

    st.markdown("---")

    # Events table below map
    st.subheader("Event Details")
    render_event_table(events_df)


def fetch_events_with_location(
    start_date: date,
    end_date: date,
    min_mentions: int = 5,
    event_groups: list[str] | None = None,
    country_code: str | None = None,
    limit: int = 500,
) -> pd.DataFrame:
    """Fetch events that have lat/long coordinates."""
    from src.config.constants import EVENT_GROUPS

    with get_session() as session:
        query = session.query(Event).filter(
            Event.event_date >= start_date,
            Event.event_date <= end_date,
            Event.action_geo_lat.isnot(None),
            Event.action_geo_long.isnot(None),
            Event.num_mentions >= min_mentions,
        )

        # Filter by event groups
        if event_groups:
            codes = []
            for group in event_groups:
                codes.extend(EVENT_GROUPS.get(group, []))
            if codes:
                query = query.filter(Event.event_root_code.in_(codes))

        # Filter by country
        if country_code:
            query = query.filter(Event.action_geo_country_code == country_code)

        events = query.order_by(
            Event.num_mentions.desc(),
        ).limit(limit).all()

    if not events:
        return pd.DataFrame()

    rows = []
    for e in events:
        rows.append({
            "id": e.id,
            "date": e.event_date,
            "lat": e.action_geo_lat,
            "lon": e.action_geo_long,
            "location": e.action_geo_name or e.action_geo_country_code,
            "country": e.action_geo_country_code,
            "event_code": e.event_root_code,
            "event_type": CAMEO_CATEGORIES.get(str(e.event_root_code).zfill(2), "Unknown"),
            "event_group": get_event_group(e.event_root_code),
            "actor1": e.actor1_name or e.actor1_code or "-",
            "actor2": e.actor2_name or e.actor2_code or "-",
            "goldstein": e.goldstein_scale or 0,
            "mentions": e.num_mentions or 0,
            "source_url": e.source_url,
        })

    return pd.DataFrame(rows)


def render_folium_map(events_df: pd.DataFrame):
    """Render an interactive Folium map with event markers."""
    # Create base map centered on events
    center_lat = events_df["lat"].mean()
    center_lon = events_df["lon"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=2,
        tiles="CartoDB positron",
    )

    # Color mapping by event group
    color_map = {
        "verbal_cooperation": "green",
        "material_cooperation": "darkgreen",
        "verbal_conflict": "orange",
        "material_conflict": "red",
        "violent_conflict": "darkred",
        "other": "gray",
    }

    # Add markers
    for _, row in events_df.iterrows():
        color = color_map.get(row["event_group"], "gray")

        # Popup content
        popup_html = f"""
        <div style="min-width: 200px;">
            <h4>{row['event_type']}</h4>
            <p><b>Date:</b> {row['date']}</p>
            <p><b>Location:</b> {row['location']}</p>
            <p><b>Actors:</b> {row['actor1']} → {row['actor2']}</p>
            <p><b>Goldstein:</b> {row['goldstein']:.1f}</p>
            <p><b>Mentions:</b> {row['mentions']}</p>
        </div>
        """

        # Size based on mentions (scaled)
        radius = min(max(row["mentions"] / 5, 3), 20)

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            popup=folium.Popup(popup_html, max_width=300),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            weight=1,
        ).add_to(m)

    # Add legend
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background-color: white; padding: 10px; border-radius: 5px;
                border: 2px solid gray; font-size: 12px; color: black;">
        <p style="margin: 0 0 5px 0; color: black;"><b>Event Groups</b></p>
        <p style="margin: 0; color: black;"><span style="color: green;">●</span> Verbal Cooperation</p>
        <p style="margin: 0; color: black;"><span style="color: darkgreen;">●</span> Material Cooperation</p>
        <p style="margin: 0; color: black;"><span style="color: orange;">●</span> Verbal Conflict</p>
        <p style="margin: 0; color: black;"><span style="color: red;">●</span> Material Conflict</p>
        <p style="margin: 0; color: black;"><span style="color: darkred;">●</span> Violent Conflict</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Display map
    st_folium(m, width=None, height=500, returned_objects=[])


def render_event_table(events_df: pd.DataFrame):
    """Render a table of events below the map."""
    display_df = events_df[[
        "date", "event_type", "actor1", "actor2", "location", "goldstein", "mentions"
    ]].copy()

    display_df.columns = ["Date", "Event Type", "Actor 1", "Actor 2", "Location", "Goldstein", "Mentions"]

    st.dataframe(
        display_df.sort_values("Date", ascending=False),
        column_config={
            "Date": st.column_config.DateColumn("Date", width="small"),
            "Goldstein": st.column_config.NumberColumn("Goldstein", format="%.1f"),
            "Mentions": st.column_config.NumberColumn("Mentions"),
        },
        hide_index=True,
        use_container_width=True,
        height=300,
    )
