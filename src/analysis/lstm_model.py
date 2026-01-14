"""
LSTM Model for Market Direction Prediction.

================================================================================
WHAT IS AN LSTM? (Long Short-Term Memory)
================================================================================

An LSTM is a type of Recurrent Neural Network (RNN) designed to learn patterns
in sequential data (like time series or text).

THE PROBLEM WITH BASIC RNNs:
Regular RNNs have "short memory" - they struggle to connect information from
many timesteps ago. This is called the "vanishing gradient problem":
- Gradients (learning signals) shrink as they flow backward through time
- By the time they reach early timesteps, they're nearly zero
- The network can't learn long-range dependencies

LSTM SOLUTION - THE MEMORY CELL:
LSTMs add a "memory cell" that can:
- Store information for long periods
- Selectively remember or forget
- Add new information when relevant

================================================================================
LSTM ARCHITECTURE (The Three Gates)
================================================================================

At each timestep t, an LSTM cell has three gates:

1. FORGET GATE (f_t):
   "What should we forget from the cell state?"
   - Looks at previous hidden state h_{t-1} and current input x_t
   - Outputs values between 0 (forget completely) and 1 (remember completely)
   - Formula: f_t = sigmoid(W_f · [h_{t-1}, x_t] + b_f)

2. INPUT GATE (i_t):
   "What new information should we store?"
   - Decides which values to update
   - Also creates candidate values to add
   - Formulas:
     i_t = sigmoid(W_i · [h_{t-1}, x_t] + b_i)
     C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)

3. OUTPUT GATE (o_t):
   "What should we output?"
   - Decides what parts of cell state to output
   - Formula: o_t = sigmoid(W_o · [h_{t-1}, x_t] + b_o)

CELL STATE UPDATE:
   C_t = f_t * C_{t-1} + i_t * C̃_t
         ↑ forget old    ↑ add new

HIDDEN STATE (output):
   h_t = o_t * tanh(C_t)

================================================================================
VISUAL REPRESENTATION
================================================================================

                   forget gate    input gate    output gate
                        ↓              ↓              ↓
                     ┌─────┐       ┌─────┐       ┌─────┐
    h_{t-1} ────────►│  σ  │──────│  σ  │──────│  σ  │──────►
    x_t    ────────►│     │       │     │       │     │
                     └──┬──┘       └──┬──┘       └──┬──┘
                        │             │             │
                        ↓             ↓             ↓
    C_{t-1} ────────► × ──────────► + ──────────────────────► C_t
                                    ↑                            │
                              ┌─────┴─────┐                      │
                              │   tanh    │                      │
                              └───────────┘                      ↓
                                                           ┌─────────┐
                                                           │  tanh   │
                                                           └────┬────┘
                                                                │
                                                                ↓
                                                               × ──────► h_t

================================================================================
WHY PyTorch?
================================================================================

PyTorch is one of two dominant deep learning frameworks (the other is TensorFlow).

WHY WE CHOSE PyTorch:
- More Pythonic (feels like regular Python)
- Eager execution (run immediately, easy to debug)
- Dynamic computation graphs (change network on the fly)
- Strong research community
- Clearer error messages

KEY PyTorch CONCEPTS:
1. Tensor: Like numpy array, but can run on GPU
2. nn.Module: Base class for all neural network layers
3. forward(): Defines how data flows through the network
4. backward(): Automatically computes gradients (autograd)
5. optimizer.step(): Updates weights using gradients

================================================================================
USAGE
================================================================================

    from src.analysis.lstm_model import MarketLSTM, LSTMTrainer
    from src.analysis.sequence_dataset import MarketSequenceDataset

    # Create dataset
    dataset = MarketSequenceDataset(symbol="SPY", sequence_length=20)
    train_loader, test_loader = dataset.get_dataloaders(batch_size=32)

    # Create model
    model = MarketLSTM(
        input_size=dataset.get_stats().n_features,
        hidden_size=64,
        num_layers=2,
    )

    # Train
    trainer = LSTMTrainer(model)
    history = trainer.fit(train_loader, test_loader, epochs=50)

    # Predict
    predictions = trainer.predict(test_loader)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TrainingHistory:
    """
    Records training metrics over epochs.

    Useful for:
    - Plotting learning curves
    - Diagnosing training issues
    - Comparing different hyperparameters
    """
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_accuracies: List[float] = field(default_factory=list)
    val_accuracies: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float('inf')


@dataclass
class PredictionResult:
    """Container for model predictions."""
    predictions: np.ndarray  # Binary predictions (0 or 1)
    probabilities: np.ndarray  # Probability of class 1
    accuracy: float
    f1: float
    auc: float


# =============================================================================
# LSTM MODEL
# =============================================================================

class MarketLSTM(nn.Module):
    """
    LSTM model for binary classification (market direction).

    Architecture:
    Input (sequence_length, input_size)
        ↓
    LSTM Layer(s)
        ↓
    Last Hidden State (hidden_size)
        ↓
    Fully Connected (hidden_size → 1)
        ↓
    Sigmoid
        ↓
    Output (probability of UP)

    nn.Module is PyTorch's base class for all neural networks.
    You must implement:
    - __init__: Define layers
    - forward: Define how data flows through layers
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """
        Initialize LSTM model.

        Args:
            input_size: Number of features per timestep
            hidden_size: Size of LSTM hidden state
                        (larger = more capacity, but slower and risk of overfitting)
            num_layers: Number of stacked LSTM layers
                       (more layers = can learn more complex patterns)
            dropout: Dropout probability between LSTM layers
                    (regularization to prevent overfitting)
        """
        # IMPORTANT: Must call parent's __init__
        super(MarketLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # ----- LSTM Layer -----
        # batch_first=True means input shape is (batch, sequence, features)
        # instead of (sequence, batch, features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Important! Our data is (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0,  # Only between layers
        )

        # ----- Fully Connected Output Layer -----
        # Takes LSTM hidden state → single prediction
        self.fc = nn.Linear(hidden_size, 1)

        # ----- Dropout for regularization -----
        self.dropout = nn.Dropout(dropout)

        # Log model size
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"MarketLSTM created with {n_params:,} parameters")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        This is where the magic happens! PyTorch automatically:
        - Tracks all operations for gradient computation
        - Enables GPU acceleration
        - Handles batching efficiently

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch_size,) with probabilities
        """
        # ----- LSTM Forward Pass -----
        # lstm_out: (batch, sequence_length, hidden_size) - output at each timestep
        # (h_n, c_n): Final hidden and cell states
        lstm_out, (h_n, c_n) = self.lstm(x)

        # ----- Take Last Timestep -----
        # We only care about the output at the LAST timestep
        # This contains information from the entire sequence
        #
        # Why last timestep?
        # The LSTM has seen all previous timesteps and accumulated information
        # in its hidden state. The last timestep has the most complete picture.
        last_hidden = lstm_out[:, -1, :]  # Shape: (batch, hidden_size)

        # ----- Apply Dropout -----
        last_hidden = self.dropout(last_hidden)

        # ----- Fully Connected Layer -----
        out = self.fc(last_hidden)  # Shape: (batch, 1)

        # ----- Sigmoid Activation -----
        # Converts raw output to probability between 0 and 1
        out = torch.sigmoid(out)

        # Remove extra dimension: (batch, 1) → (batch,)
        return out.squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions (alias for forward)."""
        self.eval()  # Set to evaluation mode (disables dropout)
        with torch.no_grad():  # Don't compute gradients
            return self.forward(x)


# =============================================================================
# TRAINER CLASS
# =============================================================================

class LSTMTrainer:
    """
    Handles training, evaluation, and prediction for LSTM model.

    THE TRAINING LOOP:
    1. Forward pass: x → model → prediction
    2. Compute loss: compare prediction to target
    3. Backward pass: compute gradients via backpropagation
    4. Update weights: move in direction that reduces loss
    5. Repeat for all batches (one epoch)
    6. Repeat for many epochs

    EARLY STOPPING:
    If validation loss doesn't improve for N epochs, stop training.
    This prevents overfitting to training data.
    """

    def __init__(
        self,
        model: MarketLSTM,
        learning_rate: float = 0.001,
        device: Optional[str] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: The LSTM model to train
            learning_rate: Step size for weight updates
                          (smaller = more stable but slower)
            device: 'cuda' for GPU, 'cpu' for CPU, None for auto-detect
        """
        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # Move model to device
        self.model = model.to(device)
        logger.info(f"Using device: {device}")

        # ----- Loss Function -----
        # BCELoss = Binary Cross-Entropy Loss
        # Measures how wrong probability predictions are
        # Loss = -[y * log(p) + (1-y) * log(1-p)]
        self.criterion = nn.BCELoss()

        # ----- Optimizer -----
        # Adam = Adaptive Moment Estimation
        # Smart optimization that adapts learning rate per parameter
        # Generally works well out of the box
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )

        self.history = TrainingHistory()

    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch (one pass through all training data).

        Returns:
            (average_loss, accuracy)
        """
        self.model.train()  # Set to training mode (enables dropout)

        total_loss = 0.0
        all_preds = []
        all_targets = []

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            # Move data to device (GPU if available)
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # ----- Forward Pass -----
            # Clear previous gradients
            # (PyTorch accumulates gradients by default)
            self.optimizer.zero_grad()

            # Get predictions
            predictions = self.model(X_batch)

            # ----- Compute Loss -----
            loss = self.criterion(predictions, y_batch)

            # ----- Backward Pass -----
            # Compute gradients via backpropagation
            # This is where PyTorch's autograd shines
            loss.backward()

            # ----- Gradient Clipping -----
            # Prevents exploding gradients (common in RNNs)
            # If gradients are too large, scale them down
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # ----- Update Weights -----
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            all_preds.extend((predictions > 0.5).cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_targets, all_preds)

        return avg_loss, accuracy

    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate on held-out data (no weight updates).

        Returns:
            (average_loss, accuracy)
        """
        self.model.eval()  # Set to evaluation mode (disables dropout)

        total_loss = 0.0
        all_preds = []
        all_targets = []

        # torch.no_grad() disables gradient computation
        # Faster and uses less memory since we don't need gradients for validation
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)

                total_loss += loss.item()
                all_preds.extend((predictions > 0.5).cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_targets, all_preds)

        return avg_loss, accuracy

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ) -> TrainingHistory:
        """
        Train the model.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Maximum number of training epochs
            early_stopping_patience: Stop if no improvement for this many epochs
            verbose: Print progress

        Returns:
            TrainingHistory with metrics from all epochs
        """
        if verbose:
            print(f"\n{'='*60}")
            print("LSTM TRAINING")
            print(f"{'='*60}")
            print(f"Device: {self.device}")
            print(f"Max epochs: {epochs}")
            print(f"Early stopping patience: {early_stopping_patience}")
            print(f"{'='*60}\n")

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self._train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self._validate_epoch(val_loader)

            # Record history
            self.history.train_losses.append(train_loss)
            self.history.val_losses.append(val_loss)
            self.history.train_accuracies.append(train_acc)
            self.history.val_accuracies.append(val_acc)

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.history.best_val_loss = val_loss
                self.history.best_epoch = epoch
                patience_counter = 0
                # Save best model state
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            # Print progress
            if verbose:
                improved = "✓" if patience_counter == 0 else ""
                print(
                    f"Epoch {epoch+1:3d}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Train Acc: {train_acc:.2%} | "
                    f"Val Acc: {val_acc:.2%} {improved}"
                )

            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    print(f"Best epoch was {self.history.best_epoch + 1} with val_loss={best_val_loss:.4f}")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.model.to(self.device)
            if verbose:
                print(f"\nRestored model from epoch {self.history.best_epoch + 1}")

        return self.history

    def predict(self, data_loader: DataLoader) -> PredictionResult:
        """
        Make predictions on a dataset.

        Args:
            data_loader: DataLoader with data to predict

        Returns:
            PredictionResult with predictions and metrics
        """
        self.model.eval()

        all_probs = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)

                probs = self.model(X_batch)

                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(y_batch.numpy())

        probs = np.array(all_probs)
        targets = np.array(all_targets)
        preds = (probs > 0.5).astype(int)

        return PredictionResult(
            predictions=preds,
            probabilities=probs,
            accuracy=accuracy_score(targets, preds),
            f1=f1_score(targets, preds, zero_division=0),
            auc=roc_auc_score(targets, probs) if len(np.unique(targets)) > 1 else 0.5,
        )

    def predict_sequence(self, sequence: np.ndarray) -> Tuple[int, float]:
        """
        Predict for a single sequence.

        Args:
            sequence: Array of shape (sequence_length, n_features)

        Returns:
            (prediction, probability)
        """
        self.model.eval()

        # Add batch dimension: (seq_len, features) → (1, seq_len, features)
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
        x = x.to(self.device)

        with torch.no_grad():
            prob = self.model(x).item()

        prediction = 1 if prob > 0.5 else 0
        return prediction, prob


# =============================================================================
# UTILITIES
# =============================================================================

def plot_training_history(history: TrainingHistory) -> None:
    """
    Plot training curves.

    Call this after training to visualize:
    - Training vs validation loss (look for overfitting)
    - Accuracy over time (learning progress)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    axes[0].plot(history.train_losses, label='Train')
    axes[0].plot(history.val_losses, label='Validation')
    axes[0].axvline(history.best_epoch, color='green', linestyle='--', label='Best')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()

    # Accuracy plot
    axes[1].plot(history.train_accuracies, label='Train')
    axes[1].plot(history.val_accuracies, label='Validation')
    axes[1].axvline(history.best_epoch, color='green', linestyle='--', label='Best')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Example: Train LSTM model on market data.

    Run this file directly:
        python -m src.analysis.lstm_model
    """
    import logging
    logging.basicConfig(level=logging.INFO)

    from datetime import date, timedelta
    from src.analysis.sequence_dataset import MarketSequenceDataset

    # ----- Create Dataset -----
    print("Creating dataset...")
    dataset = MarketSequenceDataset(
        symbol="SPY",
        sequence_length=20,
        start_date=date.today() - timedelta(days=365),
        end_date=date.today(),
        test_ratio=0.2,
    )

    stats = dataset.get_stats()
    print(f"\nDataset: {stats.train_size} train, {stats.test_size} test samples")
    print(f"Features: {stats.n_features}, Sequence length: {stats.sequence_length}")

    # ----- Create Model -----
    model = MarketLSTM(
        input_size=stats.n_features,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
    )

    # ----- Create Trainer -----
    trainer = LSTMTrainer(model, learning_rate=0.001)

    # ----- Train -----
    train_loader, test_loader = dataset.get_dataloaders(batch_size=32)
    history = trainer.fit(
        train_loader,
        test_loader,
        epochs=50,
        early_stopping_patience=10,
    )

    # ----- Evaluate -----
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    test_results = trainer.predict(test_loader)
    print(f"Test Accuracy: {test_results.accuracy:.2%}")
    print(f"Test F1 Score: {test_results.f1:.2%}")
    print(f"Test AUC: {test_results.auc:.3f}")

    # ----- Compare to Random -----
    print(f"\nBaseline (random): 50% accuracy, 0.500 AUC")
    improvement = (test_results.accuracy - 0.5) / 0.5 * 100
    print(f"Improvement over random: {improvement:+.1f}%")

    # ----- Plot History -----
    try:
        plot_training_history(history)
    except Exception as e:
        print(f"\nCouldn't plot (likely no display): {e}")
