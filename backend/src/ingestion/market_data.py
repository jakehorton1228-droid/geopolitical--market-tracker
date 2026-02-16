"""
Yahoo Finance Market Data Ingestion Module.

WHAT IS YFINANCE?
-----------------
yfinance is a Python library that fetches financial data from Yahoo Finance.
It provides:
- Historical daily/weekly/monthly prices (OHLCV)
- Stock info, dividends, splits
- Multiple asset classes: stocks, ETFs, commodities, currencies, crypto

WHY YAHOO FINANCE?
------------------
- Free (no API key required)
- Good coverage of global assets
- Reliable for daily data
- Easy to use Python library

SYMBOLS WE TRACK (from constants.py):
-------------------------------------
- Commodities: CL=F (Oil), GC=F (Gold), NG=F (Natural Gas), etc.
- Currencies: EURUSD=X, USDJPY=X, USDRUB=X, etc.
- ETFs: SPY (S&P 500), EEM (Emerging Markets), FXI (China), etc.
- Volatility: ^VIX
- Bonds: TLT (20yr Treasury), HYG (High Yield)

USAGE:
------
    from src.ingestion.market_data import MarketDataIngestion

    market = MarketDataIngestion()

    # Fetch data for one symbol
    df = market.fetch_symbol_data("SPY", start_date, end_date)

    # Fetch and store all tracked symbols
    results = market.ingest_all_symbols(start_date, end_date)
"""

import math
from datetime import date, datetime, timedelta
from typing import Optional
import logging

import pandas as pd
import yfinance as yf
import numpy as np

from src.config.constants import get_all_symbols, get_symbol_info
from src.db.connection import get_session
from src.db.models import MarketData
from src.db.queries import upsert_market_data, get_latest_market_date

logger = logging.getLogger(__name__)


class MarketDataIngestion:
    """
    Handles fetching and storing market data from Yahoo Finance.

    This class:
    1. Fetches OHLCV data for financial instruments
    2. Calculates daily and log returns
    3. Stores data in our database
    """

    def __init__(self, symbols: list[str] = None):
        """
        Initialize the market data ingestion handler.

        Args:
            symbols: List of ticker symbols to track. Defaults to all symbols
                    from constants.py
        """
        self.symbols = symbols or get_all_symbols()

    def fetch_symbol_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Fetch historical data for a single symbol.

        Uses yfinance to download OHLCV (Open, High, Low, Close, Volume) data.

        Args:
            symbol: Ticker symbol (e.g., "SPY", "CL=F", "EURUSD=X")
            start_date: Start of date range
            end_date: End of date range (inclusive)

        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Adj Close, Volume
            Plus calculated: daily_return, log_return
        """
        logger.info(f"Fetching {symbol} from {start_date} to {end_date}")

        # yfinance's end_date is exclusive, so add 1 day
        end_date_adj = end_date + timedelta(days=1)

        try:
            # Download data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date_adj.strftime("%Y-%m-%d"),
                interval="1d",  # Daily data
            )

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            # Reset index to make Date a column
            df = df.reset_index()

            # Standardize column names
            df = df.rename(columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            })

            # Add symbol column
            df["symbol"] = symbol

            # Handle timezone-aware dates (convert to date only)
            if hasattr(df["date"].iloc[0], 'date'):
                df["date"] = df["date"].dt.date
            else:
                df["date"] = pd.to_datetime(df["date"]).dt.date

            # Calculate returns
            df = self._calculate_returns(df)

            # Select and order columns
            columns = ["symbol", "date", "open", "high", "low", "close", "volume", "daily_return", "log_return"]
            available_columns = [c for c in columns if c in df.columns]
            df = df[available_columns]

            logger.info(f"Fetched {len(df)} rows for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    def _calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily and log returns.

        WHAT ARE RETURNS?
        -----------------
        - Daily Return: (today's close - yesterday's close) / yesterday's close
          Example: Stock goes from $100 to $102 → return = 2%

        - Log Return: ln(today's close / yesterday's close)
          Used in finance because log returns are additive across time.
          Example: ln(102/100) = 0.0198 ≈ 2%

        WHY LOG RETURNS?
        ----------------
        Log returns are preferred for statistical analysis because:
        1. They're additive: weekly return = sum of daily log returns
        2. They're more normally distributed
        3. They handle compounding correctly

        Args:
            df: DataFrame with 'close' column

        Returns:
            DataFrame with added 'daily_return' and 'log_return' columns
        """
        # Sort by date to ensure correct ordering
        df = df.sort_values("date")

        # Daily return: percentage change from previous close
        df["daily_return"] = df["close"].pct_change()

        # Log return: natural log of price ratio
        # We use shift(1) to get previous close
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # First row will have NaN returns (no previous day)
        # Replace NaN and inf with None for database
        df["daily_return"] = df["daily_return"].replace([np.inf, -np.inf], np.nan)
        df["log_return"] = df["log_return"].replace([np.inf, -np.inf], np.nan)

        return df

    def _row_to_market_data_dict(self, row: pd.Series) -> dict:
        """
        Convert a DataFrame row to a dictionary matching our MarketData model.

        Args:
            row: Single row from DataFrame

        Returns:
            Dictionary ready to create a MarketData object
        """
        def safe_float(value):
            """Convert to float, handling NaN and None."""
            if value is None:
                return None
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return None
            try:
                return float(value)
            except (ValueError, TypeError):
                return None

        def safe_int(value):
            """Convert to int, handling NaN and None."""
            if value is None:
                return None
            if isinstance(value, float) and math.isnan(value):
                return None
            try:
                return int(value)
            except (ValueError, TypeError):
                return None

        return {
            "symbol": str(row["symbol"]),
            "date": row["date"],
            "open": safe_float(row.get("open")),
            "high": safe_float(row.get("high")),
            "low": safe_float(row.get("low")),
            "close": safe_float(row.get("close")),
            "adj_close": safe_float(row.get("adj_close")),
            "volume": safe_int(row.get("volume")),
            "daily_return": safe_float(row.get("daily_return")),
            "log_return": safe_float(row.get("log_return")),
        }

    def ingest_symbol(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> int:
        """
        Fetch data for a symbol and store in database.

        Args:
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date

        Returns:
            Number of rows ingested
        """
        df = self.fetch_symbol_data(symbol, start_date, end_date)

        if df.empty:
            return 0

        with get_session() as session:
            for _, row in df.iterrows():
                data = self._row_to_market_data_dict(row)
                if data["close"] is not None:  # Must have close price
                    upsert_market_data(session, data)

        logger.info(f"Ingested {len(df)} rows for {symbol}")
        return len(df)

    def ingest_all_symbols(
        self,
        start_date: date,
        end_date: date,
        skip_existing: bool = True,
    ) -> dict:
        """
        Fetch and store data for all tracked symbols.

        Args:
            start_date: Start date
            end_date: End date
            skip_existing: If True, only fetch data newer than what's in DB

        Returns:
            Dictionary mapping symbol to row count
        """
        results = {}

        for symbol in self.symbols:
            try:
                # Optionally adjust start date based on existing data
                actual_start = start_date
                if skip_existing:
                    with get_session() as session:
                        latest = get_latest_market_date(session, symbol)
                        if latest and latest >= start_date:
                            actual_start = latest + timedelta(days=1)
                            if actual_start > end_date:
                                logger.info(f"{symbol}: Already up to date")
                                results[symbol] = 0
                                continue

                count = self.ingest_symbol(symbol, actual_start, end_date)
                results[symbol] = count

            except Exception as e:
                logger.error(f"Error ingesting {symbol}: {e}")
                results[symbol] = -1

        # Summary
        total = sum(c for c in results.values() if c > 0)
        success = sum(1 for c in results.values() if c >= 0)
        logger.info(f"Ingested {total} total rows for {success}/{len(self.symbols)} symbols")

        return results

    def get_symbol_summary(self) -> pd.DataFrame:
        """
        Get a summary of all tracked symbols.

        Returns:
            DataFrame with symbol info and data availability
        """
        summaries = []
        for symbol in self.symbols:
            info = get_symbol_info(symbol)
            if info:
                summaries.append(info)

        return pd.DataFrame(summaries)


def fetch_sample_prices(symbol: str = "SPY", days: int = 30) -> pd.DataFrame:
    """
    Quick helper to fetch recent prices for exploration.

    Args:
        symbol: Ticker symbol (default: SPY)
        days: Number of days to fetch

    Returns:
        DataFrame with price data
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    market = MarketDataIngestion()
    return market.fetch_symbol_data(symbol, start_date, end_date)
