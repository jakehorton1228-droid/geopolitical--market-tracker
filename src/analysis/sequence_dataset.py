"""
Sequence Dataset Preparation for LSTM Models.

================================================================================
WHY SEQUENCES FOR MARKET PREDICTION?
================================================================================

Markets have MEMORY. What happened yesterday affects today:
- A conflict that escalated over 3 days has more impact than a one-day event
- Sentiment trends matter (improving vs deteriorating)
- Technical patterns span multiple days (support/resistance levels)

Traditional ML (like XGBoost) treats each day independently:
    Day 1 features → Day 1 prediction
    Day 2 features → Day 2 prediction  (no connection to Day 1!)

LSTMs process sequences:
    [Day 1, Day 2, Day 3, ...Day 20] → Day 21 prediction
    The model learns patterns ACROSS days

================================================================================
WHAT IS A SEQUENCE?
================================================================================

For our market prediction:

SEQUENCE (X):
┌─────────────────────────────────────────────────────────────────┐
│ Day t-20: [return, volume, goldstein, mentions, embedding...]   │
│ Day t-19: [return, volume, goldstein, mentions, embedding...]   │
│ Day t-18: [return, volume, goldstein, mentions, embedding...]   │
│ ...                                                             │
│ Day t-1:  [return, volume, goldstein, mentions, embedding...]   │
└─────────────────────────────────────────────────────────────────┘

TARGET (y):
┌─────────────────────────────────────────────────────────────────┐
│ Day t: UP (1) or DOWN (0)                                       │
└─────────────────────────────────────────────────────────────────┘

The LSTM sees 20 days of history and predicts the next day's direction.

================================================================================
CRITICAL: TIME SERIES CROSS-VALIDATION
================================================================================

WRONG approach (what works for regular ML):
    Shuffle data randomly, then split into train/test
    ❌ This "leaks" future information into training!
    ❌ You'd be predicting 2023 prices with a model that saw 2024 data

CORRECT approach (for time series):
    Train on past data, test on future data
    ✓ Train: Jan-Oct 2023
    ✓ Test: Nov-Dec 2023

    For walk-forward validation:
    ┌──────────────────────────────────────────────────────────────┐
    │ Split 1: Train [Jan-Jun] → Test [Jul]                        │
    │ Split 2: Train [Jan-Jul] → Test [Aug]                        │
    │ Split 3: Train [Jan-Aug] → Test [Sep]                        │
    │ ...                                                          │
    └──────────────────────────────────────────────────────────────┘

================================================================================
NORMALIZATION FOR SEQUENCES
================================================================================

WRONG: Normalize each sequence independently
    Sequence 1 has mean=0, std=1
    Sequence 2 has mean=0, std=1  (different actual values!)
    ❌ Loses the absolute scale information

CORRECT: Normalize across all training data, apply same transform to test
    Compute mean/std from training period
    Apply same mean/std to both train and test sequences
    ✓ Preserves relative scale

================================================================================
USAGE
================================================================================

    from src.analysis.sequence_dataset import MarketSequenceDataset

    # Create dataset
    dataset = MarketSequenceDataset(
        symbol="SPY",
        sequence_length=20,
        start_date=date(2023, 1, 1),
        end_date=date(2024, 1, 1),
    )

    # Get PyTorch DataLoader
    train_loader, test_loader = dataset.get_dataloaders(batch_size=32)

    # Or access raw numpy arrays
    X_train, y_train, X_test, y_test = dataset.get_arrays()
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, Tuple
import logging

import numpy as np
import pandas as pd

# PyTorch for creating Dataset and DataLoader
import torch
from torch.utils.data import Dataset, DataLoader

from src.db.connection import get_session
from src.db.queries import get_events_by_date_range, get_market_data

logger = logging.getLogger(__name__)


# =============================================================================
# PYTORCH DATASET CLASS
# =============================================================================

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series sequences.

    PyTorch's Dataset class is the standard way to handle data.
    It requires implementing:
    - __len__: How many samples?
    - __getitem__: Get the i-th sample

    The DataLoader then uses this to:
    - Batch samples together
    - Optionally shuffle (but NOT for time series!)
    - Load data in parallel
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize with numpy arrays.

        Args:
            X: Sequences of shape (n_samples, sequence_length, n_features)
            y: Targets of shape (n_samples,)
        """
        # Convert to PyTorch tensors
        # float32 is standard for neural networks (faster than float64)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (sequence, target)
        """
        return self.X[idx], self.y[idx]


# =============================================================================
# MAIN SEQUENCE DATASET CLASS
# =============================================================================

@dataclass
class DatasetStats:
    """Statistics about the dataset."""
    n_samples: int
    n_features: int
    sequence_length: int
    train_size: int
    test_size: int
    class_balance: dict  # {0: count, 1: count}


class MarketSequenceDataset:
    """
    Prepares sequential data for LSTM market prediction.

    This class handles:
    1. Loading market and event data from the database
    2. Creating features for each day
    3. Building sequences of consecutive days
    4. Proper train/test splitting (respecting time order!)
    5. Normalization (fit on train, apply to test)
    6. Creating PyTorch DataLoaders
    """

    def __init__(
        self,
        symbol: str,
        sequence_length: int = 20,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        test_ratio: float = 0.2,
        include_embeddings: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            symbol: Market symbol to predict (e.g., "SPY")
            sequence_length: Number of days in each sequence
            start_date: Start of data range (default: 1 year ago)
            end_date: End of data range (default: today)
            test_ratio: Fraction of data for testing (most recent)
            include_embeddings: Whether to include text embeddings as features
        """
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.test_ratio = test_ratio
        self.include_embeddings = include_embeddings

        # Default date range: 1 year
        self.end_date = end_date or date.today()
        self.start_date = start_date or (self.end_date - timedelta(days=365))

        # Will be populated by _prepare_data()
        self.feature_names: list[str] = []
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

        # Normalization parameters (fit on training data)
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

        # Load and prepare data
        self._prepare_data()

    # =========================================================================
    # DATA PREPARATION
    # =========================================================================

    def _load_raw_data(self) -> pd.DataFrame:
        """
        Load and merge market data with event features.

        Returns a DataFrame with one row per day, containing:
        - Market features (return, volume, etc.)
        - Event features (aggregated goldstein, mentions, etc.)
        """
        # ----- Load market data -----
        with get_session() as session:
            market_data = get_market_data(
                session, self.symbol, self.start_date, self.end_date
            )

            if not market_data:
                raise ValueError(f"No market data found for {self.symbol}")

            market_df = pd.DataFrame([
                {
                    "date": m.date,
                    "close": m.close,
                    "volume": m.volume or 0,
                    "daily_return": m.daily_return or 0,
                    "log_return": m.log_return or 0,
                }
                for m in market_data
            ])

        # ----- Load event data -----
        with get_session() as session:
            events = get_events_by_date_range(
                session, self.start_date, self.end_date
            )

            if events:
                events_df = pd.DataFrame([
                    {
                        "date": e.event_date,
                        "goldstein_scale": e.goldstein_scale or 0,
                        "num_mentions": e.num_mentions or 0,
                        "avg_tone": e.avg_tone or 0,
                        "is_conflict": 1 if e.event_root_code in ["18", "19", "20"] else 0,
                        "is_cooperation": 1 if e.event_root_code in ["03", "04", "05", "06"] else 0,
                    }
                    for e in events
                ])

                # Aggregate events by date
                event_agg = events_df.groupby("date").agg({
                    "goldstein_scale": ["mean", "min", "max", "std"],
                    "num_mentions": ["sum", "mean"],
                    "avg_tone": "mean",
                    "is_conflict": "sum",
                    "is_cooperation": "sum",
                }).reset_index()

                # Flatten column names
                event_agg.columns = [
                    "date",
                    "goldstein_mean", "goldstein_min", "goldstein_max", "goldstein_std",
                    "mentions_total", "mentions_mean",
                    "avg_tone",
                    "conflict_count", "cooperation_count"
                ]

                event_agg["goldstein_std"] = event_agg["goldstein_std"].fillna(0)
            else:
                # Create empty event features if no events
                event_agg = pd.DataFrame({"date": market_df["date"]})
                for col in ["goldstein_mean", "goldstein_min", "goldstein_max",
                           "goldstein_std", "mentions_total", "mentions_mean",
                           "avg_tone", "conflict_count", "cooperation_count"]:
                    event_agg[col] = 0

        # ----- Merge -----
        merged = pd.merge(market_df, event_agg, on="date", how="left")
        merged = merged.fillna(0)
        merged = merged.sort_values("date").reset_index(drop=True)

        logger.info(f"Loaded {len(merged)} days of data for {self.symbol}")

        return merged

    def _create_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, list[str]]:
        """
        Create feature matrix from raw data.

        Features include:
        - Market features: return, volume (normalized)
        - Event features: goldstein stats, mentions, tone, event counts
        - Technical features: rolling averages, volatility

        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        features = []
        names = []

        # ----- Market Features -----
        # Log return is our primary price feature
        features.append(df["log_return"].values)
        names.append("log_return")

        # Volume (will be normalized later)
        features.append(df["volume"].values)
        names.append("volume")

        # ----- Event Features -----
        event_cols = [
            "goldstein_mean", "goldstein_min", "goldstein_max", "goldstein_std",
            "mentions_total", "mentions_mean", "avg_tone",
            "conflict_count", "cooperation_count"
        ]

        for col in event_cols:
            features.append(df[col].values)
            names.append(col)

        # ----- Technical Features (derived) -----
        # These capture patterns that LSTMs might learn anyway,
        # but providing them explicitly can help

        # 5-day rolling average return (momentum indicator)
        rolling_return = df["log_return"].rolling(window=5, min_periods=1).mean().values
        features.append(rolling_return)
        names.append("return_ma5")

        # 5-day rolling volatility (risk indicator)
        rolling_vol = df["log_return"].rolling(window=5, min_periods=1).std().fillna(0).values
        features.append(rolling_vol)
        names.append("volatility_5d")

        # 5-day rolling average mentions (attention indicator)
        rolling_mentions = df["mentions_total"].rolling(window=5, min_periods=1).mean().values
        features.append(rolling_mentions)
        names.append("mentions_ma5")

        # Stack into feature matrix
        X = np.column_stack(features)

        return X, names

    def _create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from feature matrix.

        INPUT:
            X: (n_days, n_features) - one row per day
            y: (n_days,) - target for each day

        OUTPUT:
            X_seq: (n_sequences, sequence_length, n_features)
            y_seq: (n_sequences,) - target is for day AFTER sequence ends

        EXAMPLE with sequence_length=3:
            Days:     [1, 2, 3, 4, 5, 6, 7]
            Sequence 1: X=[1,2,3], y=4's direction
            Sequence 2: X=[2,3,4], y=5's direction
            Sequence 3: X=[3,4,5], y=6's direction
            etc.
        """
        n_samples = len(X) - self.sequence_length
        n_features = X.shape[1]

        if n_samples <= 0:
            raise ValueError(
                f"Not enough data: {len(X)} days but need {self.sequence_length + 1}"
            )

        # Pre-allocate arrays
        X_seq = np.zeros((n_samples, self.sequence_length, n_features))
        y_seq = np.zeros(n_samples)

        # Build sequences
        for i in range(n_samples):
            # Sequence: days i to i+sequence_length-1
            X_seq[i] = X[i:i + self.sequence_length]
            # Target: day i+sequence_length (the next day)
            y_seq[i] = y[i + self.sequence_length]

        logger.info(
            f"Created {n_samples} sequences of length {self.sequence_length}"
        )

        return X_seq, y_seq

    def _normalize(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize sequences using training data statistics.

        CRITICAL: We fit the normalizer on TRAINING data only, then apply
        the same transformation to test data. This prevents information
        leakage from the future.

        Z-score normalization: (x - mean) / std
        - Centers data around 0
        - Scales to unit variance
        - Helps neural networks train faster and more stably

        Args:
            X_train: Training sequences (n_train, seq_len, n_features)
            X_test: Test sequences (n_test, seq_len, n_features)

        Returns:
            Normalized (X_train, X_test)
        """
        # Reshape to 2D for computing statistics
        # We want mean/std per feature across ALL timesteps
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        n_features = X_train.shape[2]

        X_train_flat = X_train.reshape(-1, n_features)

        # Compute statistics from training data only
        self._mean = X_train_flat.mean(axis=0)
        self._std = X_train_flat.std(axis=0)

        # Avoid division by zero (constant features)
        self._std[self._std < 1e-8] = 1.0

        # Normalize both sets using training statistics
        X_train_norm = (X_train_flat - self._mean) / self._std
        X_train_norm = X_train_norm.reshape(n_train, self.sequence_length, n_features)

        X_test_flat = X_test.reshape(-1, n_features)
        X_test_norm = (X_test_flat - self._mean) / self._std
        X_test_norm = X_test_norm.reshape(n_test, self.sequence_length, n_features)

        return X_train_norm, X_test_norm

    def _prepare_data(self) -> None:
        """
        Main data preparation pipeline.

        Steps:
        1. Load raw data
        2. Create features
        3. Create target variable
        4. Split train/test (by time, not random!)
        5. Create sequences
        6. Normalize
        """
        # Step 1: Load data
        df = self._load_raw_data()

        # Step 2: Create features
        X, self.feature_names = self._create_features(df)

        # Step 3: Create target (1 = UP, 0 = DOWN)
        y = (df["log_return"] > 0).astype(int).values

        # Step 4: Time-based train/test split
        # IMPORTANT: We don't shuffle! Test data must be AFTER training data
        split_idx = int(len(X) * (1 - self.test_ratio))

        X_train_raw = X[:split_idx]
        X_test_raw = X[split_idx:]
        y_train_raw = y[:split_idx]
        y_test_raw = y[split_idx:]

        # Step 5: Create sequences
        X_train_seq, y_train_seq = self._create_sequences(X_train_raw, y_train_raw)
        X_test_seq, y_test_seq = self._create_sequences(X_test_raw, y_test_raw)

        # Step 6: Normalize
        self.X_train, self.X_test = self._normalize(X_train_seq, X_test_seq)
        self.y_train = y_train_seq
        self.y_test = y_test_seq

        logger.info(
            f"Dataset ready: {len(self.X_train)} train, {len(self.X_test)} test samples"
        )

    # =========================================================================
    # DATA ACCESS METHODS
    # =========================================================================

    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get raw numpy arrays for train and test sets.

        Returns:
            (X_train, y_train, X_test, y_test)
        """
        return self.X_train, self.y_train, self.X_test, self.y_test

    def get_dataloaders(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Get PyTorch DataLoaders for training.

        DataLoader benefits:
        - Automatic batching
        - Optional shuffling (we disable for time series)
        - Parallel data loading
        - Memory-efficient iteration

        Args:
            batch_size: Samples per batch (32 is a good default)
            num_workers: Parallel data loading workers (0 = main thread)

        Returns:
            (train_loader, test_loader)
        """
        train_dataset = TimeSeriesDataset(self.X_train, self.y_train)
        test_dataset = TimeSeriesDataset(self.X_test, self.y_test)

        # IMPORTANT: shuffle=False for time series!
        # Shuffling would mix up the temporal order
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,  # NO SHUFFLING for time series
            num_workers=num_workers,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        return train_loader, test_loader

    def get_stats(self) -> DatasetStats:
        """Get statistics about the dataset."""
        train_classes, train_counts = np.unique(self.y_train, return_counts=True)
        class_balance = dict(zip(train_classes.astype(int), train_counts))

        return DatasetStats(
            n_samples=len(self.X_train) + len(self.X_test),
            n_features=self.X_train.shape[2],
            sequence_length=self.sequence_length,
            train_size=len(self.X_train),
            test_size=len(self.X_test),
            class_balance=class_balance,
        )

    def get_sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single batch for testing model architecture.

        Useful for:
        - Checking input/output shapes
        - Debugging model forward pass
        - Quick sanity checks
        """
        train_loader, _ = self.get_dataloaders(batch_size=8)
        X_batch, y_batch = next(iter(train_loader))
        return X_batch, y_batch


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Example: Create sequence dataset and inspect it.

    Run this file directly:
        python -m src.analysis.sequence_dataset
    """
    import logging
    logging.basicConfig(level=logging.INFO)

    from datetime import date, timedelta

    # Create dataset
    print("Creating sequence dataset for SPY...")
    print("="*60)

    dataset = MarketSequenceDataset(
        symbol="SPY",
        sequence_length=20,
        start_date=date.today() - timedelta(days=365),
        end_date=date.today(),
        test_ratio=0.2,
    )

    # Get statistics
    stats = dataset.get_stats()
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {stats.n_samples}")
    print(f"  Training samples: {stats.train_size}")
    print(f"  Test samples: {stats.test_size}")
    print(f"  Sequence length: {stats.sequence_length}")
    print(f"  Features per timestep: {stats.n_features}")
    print(f"  Class balance (train): {stats.class_balance}")

    # Get a sample batch
    print(f"\nSample Batch:")
    X_batch, y_batch = dataset.get_sample_batch()
    print(f"  X shape: {X_batch.shape}  # (batch, sequence_length, features)")
    print(f"  y shape: {y_batch.shape}  # (batch,)")

    # Show feature names
    print(f"\nFeatures ({len(dataset.feature_names)}):")
    for i, name in enumerate(dataset.feature_names):
        print(f"  {i+1}. {name}")

    # Get DataLoaders
    print(f"\nDataLoaders:")
    train_loader, test_loader = dataset.get_dataloaders(batch_size=32)
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
