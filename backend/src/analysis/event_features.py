"""
Event-Centered ML Feature Pipeline.

This is a reframing of the ML task. Instead of predicting "will today's
market move be significant" across every (symbol, day) pair (which has
terrible signal-to-noise and led to target leakage), we predict:

    "Given a SIGNIFICANT geopolitical event occurred, will the affected
     asset move UP over the next N trading days?"

Why this is a better problem:
- Training data is restricted to events with real signal (high Goldstein
  magnitude + high media coverage)
- Target uses FUTURE returns relative to the event, avoiding same-day
  contamination
- Each sample has a clear trigger (a specific event) and a specific
  affected asset (derived from country → asset mapping)
- Achievable performance is 0.60-0.75 AUC on event-conditional prediction
  (vs 0.50-0.55 on all-days direction prediction)

DATASET CONSTRUCTION:
1. Filter events: |Goldstein| >= 5 AND num_mentions >= 1000 (significant)
2. For each event, look up affected assets via COUNTRY_ASSET_MAP
3. For each (event, asset) pair, compute target = 1 if cumulative return
   over [event_date+1, event_date+horizon] > 0, else 0
4. Features: event metrics, event category flags, asset context (lagged
   so no leakage), recent sentiment in the event's country

USAGE:
------
    from src.analysis.event_features import EventFeaturePipeline

    pipeline = EventFeaturePipeline()
    data = pipeline.build_dataset()
    X_train, y_train = data["X_train"], data["y_train"]
"""

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select, func, or_

from src.config.constants import COUNTRY_ASSET_MAP, EVENT_GROUPS
from src.db.connection import get_session
from src.db.models import Event, MarketData, NewsHeadline

logger = logging.getLogger(__name__)

# --- Tunable parameters ----------------------------------------------------

# Event significance thresholds. Lower = more data, more noise.
MIN_GOLDSTEIN_MAGNITUDE = 5.0
MIN_MENTIONS = 1000

# How far ahead to evaluate the asset's reaction to the event
HORIZON_DAYS = 3

# Train/val/test split (time-series, chronological)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
# TEST_RATIO = 0.15 remainder

DEFAULT_START = date(2016, 1, 1)


class EventFeaturePipeline:
    """Produces an event-centered dataset for ML training."""

    def build_dataset(
        self,
        start_date: date = DEFAULT_START,
        end_date: Optional[date] = None,
        horizon_days: int = HORIZON_DAYS,
    ) -> dict:
        """Build the full (train, val, test) event-centered dataset.

        Returns dict with:
            X_train, X_val, X_test: numpy arrays of features
            y_train, y_val, y_test: numpy arrays of binary targets
            feature_names: ordered list of feature column names
            n_events: total events used
            n_samples: total (event, asset) training rows
        """
        if end_date is None:
            end_date = date.today()

        logger.info(f"Building event-centered dataset: {start_date} to {end_date}")

        # 1. Fetch significant events
        events_df = self._fetch_significant_events(start_date, end_date)
        if events_df.empty:
            raise ValueError("No significant events found in date range")
        logger.info(f"Found {len(events_df)} significant events")

        # 2. Expand to (event, asset) pairs via country mapping
        pairs_df = self._expand_to_event_asset_pairs(events_df)
        if pairs_df.empty:
            raise ValueError("No events matched to any tracked assets")
        logger.info(f"Expanded to {len(pairs_df)} (event, asset) pairs")

        # 3. Compute targets from future returns
        pairs_df = self._add_targets(pairs_df, horizon_days)
        pairs_df = pairs_df.dropna(subset=["target"])
        logger.info(f"After target computation: {len(pairs_df)} samples")

        # 4. Add asset context features (lagged, no leakage)
        pairs_df = self._add_asset_context(pairs_df)

        # 5. Add event category flags
        pairs_df = self._add_event_category_flags(pairs_df)

        # 6. Add asset identity (one-hot of the tracked symbol)
        pairs_df = self._add_asset_identity(pairs_df)

        # 7. Drop rows with missing context (early dates don't have lookback)
        pairs_df = pairs_df.dropna().reset_index(drop=True)
        logger.info(f"Final dataset: {len(pairs_df)} samples, target balance: {pairs_df['target'].mean():.1%} positive")

        # 8. Split chronologically
        return self._chronological_split(pairs_df)

    # ------------------------------------------------------------------
    # Step 1: fetch significant events
    # ------------------------------------------------------------------

    def _fetch_significant_events(
        self, start_date: date, end_date: date,
    ) -> pd.DataFrame:
        """Pull events with high Goldstein magnitude AND high media coverage."""
        with get_session() as session:
            rows = session.execute(
                select(
                    Event.id,
                    Event.event_date,
                    Event.event_root_code,
                    Event.actor1_country_code,
                    Event.actor2_country_code,
                    Event.action_geo_country_code,
                    Event.goldstein_scale,
                    Event.num_mentions,
                    Event.num_sources,
                    Event.avg_tone,
                ).where(
                    Event.event_date >= start_date,
                    Event.event_date <= end_date,
                    func.abs(Event.goldstein_scale) >= MIN_GOLDSTEIN_MAGNITUDE,
                    Event.num_mentions >= MIN_MENTIONS,
                    Event.event_root_code.isnot(None),
                )
            ).all()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=[
            "event_id", "event_date", "event_root_code",
            "actor1_country", "actor2_country", "action_country",
            "goldstein_scale", "num_mentions", "num_sources", "avg_tone",
        ])
        df["event_date"] = pd.to_datetime(df["event_date"]).dt.date
        df["goldstein_scale"] = df["goldstein_scale"].fillna(0).astype(float)
        df["num_mentions"] = df["num_mentions"].fillna(0).astype(int)
        df["num_sources"] = df["num_sources"].fillna(0).astype(int)
        df["avg_tone"] = df["avg_tone"].fillna(0).astype(float)
        return df

    # ------------------------------------------------------------------
    # Step 2: event → affected asset expansion
    # ------------------------------------------------------------------

    def _expand_to_event_asset_pairs(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """For each event, generate one row per affected asset.

        An event is mapped to assets whose sensitive countries match any
        of the event's actors (actor1, actor2) or action geography.
        """
        expanded_rows = []
        for _, row in events_df.iterrows():
            # Gather all countries associated with this event
            countries = {
                c for c in [
                    row["actor1_country"],
                    row["actor2_country"],
                    row["action_country"],
                ]
                if c and c in COUNTRY_ASSET_MAP
            }

            if not countries:
                continue

            # Collect all assets affected by those countries (deduped)
            affected_symbols = set()
            for country in countries:
                affected_symbols.update(COUNTRY_ASSET_MAP[country])

            # One row per (event, symbol)
            primary_country = next(iter(countries))
            for symbol in affected_symbols:
                expanded_rows.append({
                    **row.to_dict(),
                    "symbol": symbol,
                    "primary_country": primary_country,
                })

        if not expanded_rows:
            return pd.DataFrame()

        return pd.DataFrame(expanded_rows)

    # ------------------------------------------------------------------
    # Step 3: compute future-return targets
    # ------------------------------------------------------------------

    def _add_targets(self, pairs_df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
        """For each (event, asset) pair, compute the target and features.

        Target: did log_return cumulated over [event_date+1, event_date+horizon_days] > 0?
        """
        # Fetch all needed market data in one go — one query per symbol
        symbols = pairs_df["symbol"].unique().tolist()
        earliest = pd.to_datetime(pairs_df["event_date"].min()) - pd.Timedelta(days=30)
        latest = pd.to_datetime(pairs_df["event_date"].max()) + pd.Timedelta(days=horizon_days + 5)

        market_by_symbol = {}
        with get_session() as session:
            for symbol in symbols:
                rows = session.execute(
                    select(
                        MarketData.date,
                        MarketData.close,
                        MarketData.log_return,
                        MarketData.volume,
                    ).where(
                        MarketData.symbol == symbol,
                        MarketData.date >= earliest.date(),
                        MarketData.date <= latest.date(),
                    ).order_by(MarketData.date)
                ).all()
                if rows:
                    market_df = pd.DataFrame(rows, columns=["date", "close", "log_return", "volume"])
                    market_df["date"] = pd.to_datetime(market_df["date"]).dt.date
                    market_df["close"] = market_df["close"].astype(float)
                    market_df["log_return"] = market_df["log_return"].fillna(0).astype(float)
                    market_by_symbol[symbol] = market_df

        # Compute the target for each row
        targets = []
        for _, row in pairs_df.iterrows():
            symbol = row["symbol"]
            event_date = row["event_date"]

            market_df = market_by_symbol.get(symbol)
            if market_df is None:
                targets.append(None)
                continue

            # Find rows strictly AFTER the event date
            future = market_df[market_df["date"] > event_date].head(horizon_days)
            if len(future) < horizon_days:
                targets.append(None)
                continue

            cum_return = future["log_return"].sum()
            targets.append(1 if cum_return > 0 else 0)

        pairs_df = pairs_df.copy()
        pairs_df["target"] = targets
        pairs_df["_market_cache"] = [market_by_symbol.get(s) for s in pairs_df["symbol"]]
        return pairs_df

    # ------------------------------------------------------------------
    # Step 4: lagged asset context (no leakage)
    # ------------------------------------------------------------------

    def _add_asset_context(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """Add recent asset volatility, momentum, volume trend.

        ALL features use data from dates STRICTLY BEFORE event_date so
        there's no leakage from the event window into the features.
        """
        vol_20d = []
        mom_5d = []
        mom_20d = []
        ret_5d_mean = []
        volume_ratio = []

        for _, row in pairs_df.iterrows():
            market_df = row["_market_cache"]
            event_date = row["event_date"]

            if market_df is None:
                vol_20d.append(np.nan)
                mom_5d.append(np.nan)
                mom_20d.append(np.nan)
                ret_5d_mean.append(np.nan)
                volume_ratio.append(np.nan)
                continue

            # Use ONLY pre-event history
            prior = market_df[market_df["date"] < event_date]
            if len(prior) < 20:
                vol_20d.append(np.nan)
                mom_5d.append(np.nan)
                mom_20d.append(np.nan)
                ret_5d_mean.append(np.nan)
                volume_ratio.append(np.nan)
                continue

            recent_20 = prior.tail(20)
            recent_5 = prior.tail(5)

            vol_20d.append(recent_20["log_return"].std())
            ret_5d_mean.append(recent_5["log_return"].mean())

            # Momentum: price change over N prior days
            prior_close = prior["close"].iloc[-1]
            close_5_ago = prior["close"].iloc[-5] if len(prior) >= 5 else prior_close
            close_20_ago = prior["close"].iloc[-20] if len(prior) >= 20 else prior_close
            mom_5d.append((prior_close - close_5_ago) / close_5_ago)
            mom_20d.append((prior_close - close_20_ago) / close_20_ago)

            # Volume ratio (latest day's volume vs 20-day mean)
            if recent_20["volume"].notna().any():
                vol_mean = recent_20["volume"].mean()
                vol_latest = prior["volume"].iloc[-1]
                volume_ratio.append(vol_latest / vol_mean if vol_mean > 0 else 1.0)
            else:
                volume_ratio.append(1.0)

        pairs_df = pairs_df.copy()
        pairs_df["prior_volatility_20d"] = vol_20d
        pairs_df["prior_momentum_5d"] = mom_5d
        pairs_df["prior_momentum_20d"] = mom_20d
        pairs_df["prior_return_5d_mean"] = ret_5d_mean
        pairs_df["prior_volume_ratio"] = volume_ratio
        return pairs_df

    # ------------------------------------------------------------------
    # Step 5: event category flags
    # ------------------------------------------------------------------

    def _add_event_category_flags(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """Convert CAMEO root code to boolean category flags."""
        pairs_df = pairs_df.copy()
        root_codes = pairs_df["event_root_code"].astype(str).str.zfill(2)

        for group_name, codes in EVENT_GROUPS.items():
            pairs_df[f"is_{group_name}"] = root_codes.isin(codes).astype(int)

        pairs_df["is_conflict_signal"] = (
            pairs_df["goldstein_scale"] < -MIN_GOLDSTEIN_MAGNITUDE
        ).astype(int)
        pairs_df["is_cooperation_signal"] = (
            pairs_df["goldstein_scale"] > MIN_GOLDSTEIN_MAGNITUDE
        ).astype(int)
        return pairs_df

    # ------------------------------------------------------------------
    # Step 6: asset identity (label-encoded, model learns which asset)
    # ------------------------------------------------------------------

    def _add_asset_identity(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """Label-encode the symbol so the model can learn asset-specific behavior."""
        pairs_df = pairs_df.copy()
        symbols = sorted(pairs_df["symbol"].unique())
        symbol_to_idx = {s: i for i, s in enumerate(symbols)}
        pairs_df["symbol_id"] = pairs_df["symbol"].map(symbol_to_idx)
        return pairs_df

    # ------------------------------------------------------------------
    # Step 7: chronological split
    # ------------------------------------------------------------------

    FEATURE_COLS = [
        # Event magnitude
        "goldstein_scale",
        "num_mentions",
        "num_sources",
        "avg_tone",
        # Event category flags
        "is_verbal_cooperation",
        "is_material_cooperation",
        "is_verbal_conflict",
        "is_material_conflict",
        "is_violent_conflict",
        "is_conflict_signal",
        "is_cooperation_signal",
        # Asset identity
        "symbol_id",
        # Lagged asset context (pre-event history, no leakage)
        "prior_volatility_20d",
        "prior_momentum_5d",
        "prior_momentum_20d",
        "prior_return_5d_mean",
        "prior_volume_ratio",
    ]

    def _chronological_split(self, df: pd.DataFrame) -> dict:
        """Split dataset by event_date to prevent temporal leakage."""
        df = df.sort_values("event_date").reset_index(drop=True)

        n = len(df)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]

        logger.info(
            f"Split: train={len(train)} ({train['event_date'].min()} to {train['event_date'].max()}), "
            f"val={len(val)} ({val['event_date'].min()} to {val['event_date'].max()}), "
            f"test={len(test)} ({test['event_date'].min()} to {test['event_date'].max()})"
        )

        feature_cols = self.FEATURE_COLS
        return {
            "X_train": train[feature_cols].values.astype(np.float32),
            "X_val": val[feature_cols].values.astype(np.float32),
            "X_test": test[feature_cols].values.astype(np.float32),
            "y_train": train["target"].values.astype(np.int64),
            "y_val": val["target"].values.astype(np.int64),
            "y_test": test["target"].values.astype(np.int64),
            "feature_names": feature_cols,
            "n_events": df["event_id"].nunique(),
            "n_samples": len(df),
            "symbols_train": train["symbol"].values,
            "symbols_val": val["symbol"].values,
            "symbols_test": test["symbol"].values,
            "dates_train": train["event_date"].values,
            "dates_val": val["event_date"].values,
            "dates_test": test["event_date"].values,
        }
