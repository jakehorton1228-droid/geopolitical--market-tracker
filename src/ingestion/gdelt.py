"""
GDELT Event Data Ingestion Module.

WHAT IS GDELT?
--------------
GDELT (Global Database of Events, Language, and Tone) monitors news media
worldwide and extracts structured event data. It answers: "Who did what to
whom, when, and where?"

Every day, GDELT processes news in 100+ languages and publishes a CSV file
containing all detected events.

DATA FORMAT:
-----------
GDELT files are tab-separated with 58 columns. Key columns include:
- GlobalEventID: Unique identifier for the event
- Actor1/Actor2: The parties involved (countries, organizations, etc.)
- EventCode: CAMEO code describing the type of action (01-20)
- GoldsteinScale: Conflict/cooperation score (-10 to +10)
- NumMentions: Media coverage indicator

URL PATTERN:
-----------
Daily files: http://data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip
Example: http://data.gdeltproject.org/events/20240115.export.CSV.zip

USAGE:
------
    from src.ingestion.gdelt import GDELTIngestion

    gdelt = GDELTIngestion()

    # Fetch events for a specific date
    events = gdelt.fetch_events_for_date(date(2024, 1, 15))

    # Fetch and store in database
    count = gdelt.ingest_date(date(2024, 1, 15))
    print(f"Ingested {count} events")
"""

import io
import zipfile
from datetime import date, datetime
from typing import Generator
import logging

import pandas as pd
import requests

from src.config.settings import GDELT_EVENTS_URL, GDELT_MIN_MENTIONS
from src.db.connection import get_session
from src.db.models import Event
from src.db.queries import upsert_event

# Set up logging so we can see what's happening
logger = logging.getLogger(__name__)


# GDELT column definitions
# The CSV has 58 columns - these are the indices for the ones we care about
GDELT_COLUMNS = {
    "GlobalEventID": 0,
    "Day": 1,  # YYYYMMDD format
    "Actor1Code": 5,
    "Actor1Name": 6,
    "Actor1CountryCode": 7,
    "Actor1Type1Code": 12,
    "Actor2Code": 15,
    "Actor2Name": 16,
    "Actor2CountryCode": 17,
    "Actor2Type1Code": 22,
    "IsRootEvent": 25,
    "EventCode": 26,  # Full CAMEO code
    "EventBaseCode": 27,  # Base CAMEO code
    "EventRootCode": 28,  # Root CAMEO code (01-20)
    "GoldsteinScale": 30,
    "NumMentions": 31,
    "NumSources": 32,
    "NumArticles": 33,
    "AvgTone": 34,
    "ActionGeo_CountryCode": 51,
    "ActionGeo_FullName": 50,
    "ActionGeo_Lat": 53,
    "ActionGeo_Long": 54,
    "SOURCEURL": 57,
}

# Full list of all 58 column names for pandas
ALL_COLUMN_NAMES = [
    "GlobalEventID", "Day", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources", "NumArticles",
    "AvgTone", "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code", "Actor1Geo_Lat", "Actor1Geo_Long",
    "Actor1Geo_FeatureID", "Actor2Geo_Type", "Actor2Geo_FullName",
    "Actor2Geo_CountryCode", "Actor2Geo_ADM1Code", "Actor2Geo_ADM2Code",
    "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code", "ActionGeo_ADM2Code", "ActionGeo_Lat", "ActionGeo_Long",
    "DATEADDED", "SOURCEURL",
]


class GDELTIngestion:
    """
    Handles fetching and processing GDELT event data.

    This class:
    1. Downloads daily GDELT CSV files
    2. Parses them into structured data
    3. Filters for significant events
    4. Stores them in our database
    """

    def __init__(self, min_mentions: int = None):
        """
        Initialize the GDELT ingestion handler.

        Args:
            min_mentions: Minimum number of media mentions to consider an event
                         significant. Defaults to GDELT_MIN_MENTIONS from settings.
        """
        self.base_url = GDELT_EVENTS_URL
        self.min_mentions = min_mentions or GDELT_MIN_MENTIONS

    def _build_url(self, event_date: date) -> str:
        """
        Build the URL for a specific date's GDELT file.

        Args:
            event_date: The date to fetch events for

        Returns:
            URL string like "http://data.gdeltproject.org/events/20240115.export.CSV.zip"
        """
        date_str = event_date.strftime("%Y%m%d")
        return f"{self.base_url}/{date_str}.export.CSV.zip"

    def _download_and_extract(self, url: str) -> str:
        """
        Download a GDELT zip file and extract the CSV content.

        GDELT files are distributed as zip archives containing a single CSV.
        This method downloads the zip and extracts the text content.

        Args:
            url: URL of the GDELT zip file

        Returns:
            CSV content as a string

        Raises:
            requests.HTTPError: If download fails
            zipfile.BadZipFile: If file is corrupted
        """
        logger.info(f"Downloading {url}")

        # Download the zip file
        response = requests.get(url, timeout=60)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Extract CSV from zip (it's in-memory, not saved to disk)
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            # GDELT zips contain a single CSV file
            csv_filename = zf.namelist()[0]
            csv_content = zf.read(csv_filename).decode("utf-8")

        logger.info(f"Downloaded and extracted {len(csv_content):,} bytes")
        return csv_content

    def _parse_csv(self, csv_content: str) -> pd.DataFrame:
        """
        Parse GDELT CSV content into a pandas DataFrame.

        GDELT CSV files are:
        - Tab-separated (not comma-separated)
        - No header row (column names not included)
        - 58 columns per row

        Args:
            csv_content: Raw CSV text

        Returns:
            DataFrame with named columns
        """
        df = pd.read_csv(
            io.StringIO(csv_content),
            sep="\t",  # Tab-separated
            header=None,  # No header row
            names=ALL_COLUMN_NAMES,  # We provide the column names
            dtype=str,  # Read everything as strings initially (safer)
            on_bad_lines="skip",  # Skip malformed rows
        )

        logger.info(f"Parsed {len(df):,} total events")
        return df

    def _filter_significant_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to keep only significant events.

        We filter out noise by requiring:
        1. Minimum number of media mentions (indicates real news coverage)
        2. Valid event root code (01-20)
        3. At least one identifiable actor

        This typically reduces millions of events to thousands of significant ones.

        Args:
            df: Raw DataFrame with all events

        Returns:
            Filtered DataFrame with significant events only
        """
        # Convert NumMentions to numeric for filtering
        df["NumMentions"] = pd.to_numeric(df["NumMentions"], errors="coerce").fillna(0)

        # Filter criteria
        filtered = df[
            (df["NumMentions"] >= self.min_mentions) &  # Minimum media coverage
            (df["EventRootCode"].notna()) &  # Has event type
            (df["EventRootCode"] != "") &
            (
                (df["Actor1Code"].notna() & (df["Actor1Code"] != "")) |
                (df["Actor2Code"].notna() & (df["Actor2Code"] != ""))
            )  # At least one actor identified
        ].copy()

        logger.info(f"Filtered to {len(filtered):,} significant events (min {self.min_mentions} mentions)")
        return filtered

    def _row_to_event_dict(self, row: pd.Series) -> dict:
        """
        Convert a DataFrame row to a dictionary matching our Event model.

        This maps GDELT column names to our database schema.

        Args:
            row: Single row from GDELT DataFrame

        Returns:
            Dictionary ready to create an Event object
        """
        # Parse the date from YYYYMMDD format
        try:
            event_date = datetime.strptime(str(row["Day"]), "%Y%m%d").date()
        except (ValueError, TypeError):
            event_date = None

        # Helper to safely convert to float
        def safe_float(value):
            try:
                return float(value) if pd.notna(value) and value != "" else None
            except (ValueError, TypeError):
                return None

        # Helper to safely convert to int
        def safe_int(value):
            try:
                return int(float(value)) if pd.notna(value) and value != "" else None
            except (ValueError, TypeError):
                return None

        # Helper to clean string values
        def clean_str(value, max_length=None):
            if pd.isna(value) or value == "":
                return None
            s = str(value).strip()
            if max_length:
                s = s[:max_length]
            return s if s else None

        return {
            "global_event_id": clean_str(row["GlobalEventID"], 50),
            "event_date": event_date,
            "event_root_code": clean_str(row["EventRootCode"], 2),
            "event_base_code": clean_str(row["EventBaseCode"], 4),
            "event_code": clean_str(row["EventCode"], 10),
            "actor1_code": clean_str(row["Actor1Code"], 10),
            "actor1_name": clean_str(row["Actor1Name"], 255),
            "actor1_country_code": clean_str(row["Actor1CountryCode"], 3),
            "actor1_type": clean_str(row["Actor1Type1Code"], 3),
            "actor2_code": clean_str(row["Actor2Code"], 10),
            "actor2_name": clean_str(row["Actor2Name"], 255),
            "actor2_country_code": clean_str(row["Actor2CountryCode"], 3),
            "actor2_type": clean_str(row["Actor2Type1Code"], 3),
            "is_root_event": row["IsRootEvent"] == "1",
            "goldstein_scale": safe_float(row["GoldsteinScale"]),
            "num_mentions": safe_int(row["NumMentions"]),
            "num_sources": safe_int(row["NumSources"]),
            "num_articles": safe_int(row["NumArticles"]),
            "avg_tone": safe_float(row["AvgTone"]),
            "action_geo_country_code": clean_str(row["ActionGeo_CountryCode"], 3),
            "action_geo_name": clean_str(row["ActionGeo_FullName"], 255),
            "action_geo_lat": safe_float(row["ActionGeo_Lat"]),
            "action_geo_long": safe_float(row["ActionGeo_Long"]),
            "source_url": clean_str(row["SOURCEURL"]),
        }

    def fetch_events_for_date(self, event_date: date) -> list[dict]:
        """
        Fetch and parse GDELT events for a specific date.

        This is the main method for getting event data without storing it.
        Useful for exploration or one-off analysis.

        Args:
            event_date: Date to fetch events for

        Returns:
            List of event dictionaries
        """
        url = self._build_url(event_date)

        try:
            csv_content = self._download_and_extract(url)
            df = self._parse_csv(csv_content)
            df_filtered = self._filter_significant_events(df)

            events = [self._row_to_event_dict(row) for _, row in df_filtered.iterrows()]
            return events

        except requests.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"No GDELT data available for {event_date}")
                return []
            raise

    def ingest_date(self, event_date: date) -> int:
        """
        Fetch events for a date and store them in the database.

        This is the main method for production ingestion.
        Uses upsert to handle re-runs safely (won't create duplicates).

        Args:
            event_date: Date to ingest

        Returns:
            Number of events ingested
        """
        events = self.fetch_events_for_date(event_date)

        if not events:
            return 0

        with get_session() as session:
            for event_data in events:
                if event_data["global_event_id"]:  # Skip if no ID
                    upsert_event(session, event_data)

        logger.info(f"Ingested {len(events)} events for {event_date}")
        return len(events)

    def ingest_date_range(self, start_date: date, end_date: date) -> dict:
        """
        Ingest events for a range of dates.

        Args:
            start_date: First date to ingest (inclusive)
            end_date: Last date to ingest (inclusive)

        Returns:
            Dictionary with counts per date
        """
        from datetime import timedelta

        results = {}
        current_date = start_date

        while current_date <= end_date:
            try:
                count = self.ingest_date(current_date)
                results[current_date] = count
                logger.info(f"{current_date}: {count} events")
            except Exception as e:
                logger.error(f"{current_date}: Error - {e}")
                results[current_date] = -1

            current_date += timedelta(days=1)

        total = sum(c for c in results.values() if c > 0)
        logger.info(f"Total ingested: {total} events over {len(results)} days")
        return results


# Convenience function for quick testing
def fetch_sample_events(event_date: date = None, limit: int = 5) -> list[dict]:
    """
    Fetch a few sample events for testing/exploration.

    Args:
        event_date: Date to fetch (defaults to 7 days ago)
        limit: Number of events to return

    Returns:
        List of event dictionaries
    """
    from datetime import timedelta

    if event_date is None:
        event_date = date.today() - timedelta(days=7)

    gdelt = GDELTIngestion()
    events = gdelt.fetch_events_for_date(event_date)
    return events[:limit]
