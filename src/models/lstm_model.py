"""
LSTM Model for Order Book Imbalance Forecasting

Sequence-to-classification model that predicts short-term price direction
based on order book microstructure features.

Architecture:
- Input: Sequence of order book features
- LSTM layers with dropout
- Dense classification head
- Output: Probability distribution over {down, flat, up}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


class OrderBookLSTM(nn.Module):
    """
    LSTM model for order book imbalance prediction.

    Predicts future price movement (up/down/flat) based on
    sequences of order book features.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        """
        Initialize LSTM model.

        Args:
            input_size: Number of input features
            hidden_size: Hidden state dimension
            num_layers: Number of LSTM layers
            num_classes: Number of output classes (3 for up/down/flat)
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(OrderBookLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output multiplier for bidirectional
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        # Classification head
        self.fc1 = nn.Linear(lstm_output_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(lstm_output_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
            hidden: Optional initial hidden state

        Returns:
            Tuple of (logits, hidden_state)
        """
        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)

        # Take the last output
        last_output = lstm_out[:, -1, :]

        # Layer normalization
        last_output = self.layer_norm(last_output)

        # Classification head
        out = self.dropout(last_output)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        logits = self.fc2(out)

        return logits, hidden

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.

        Args:
            x: Input tensor

        Returns:
            Probability distribution over classes
        """
        with torch.no_grad():
            logits, _ = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels.

        Args:
            x: Input tensor

        Returns:
            Predicted class indices
        """
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)


class AttentionLSTM(nn.Module):
    """
    LSTM with attention mechanism for order book forecasting.

    Attention allows the model to focus on important time steps
    in the sequence.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        attention_heads: int = 4,
    ):
        super(AttentionLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)

        # Layer norms
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention."""
        # LSTM encoding
        lstm_out, _ = self.lstm(x)
        lstm_out = self.ln1(lstm_out)

        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.ln2(attn_out)

        # Global average pooling over sequence
        pooled = torch.mean(attn_out, dim=1)

        # Classification
        out = self.dropout(pooled)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        logits = self.fc2(out)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)


# Utility functions for model creation


def create_model(model_type: str = "lstm", input_size: int = 16, **kwargs) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_type: Type of model ('lstm' or 'attention_lstm')
        input_size: Number of input features
        **kwargs: Additional model parameters

    Returns:
        PyTorch model
    """
    if model_type == "lstm":
        return OrderBookLSTM(input_size=input_size, **kwargs)
    elif model_type == "attention_lstm":
        return AttentionLSTM(input_size=input_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Example usage
if __name__ == "__main__":
    # Create model
    model = create_model(
        model_type="lstm", input_size=16, hidden_size=128, num_layers=2, dropout=0.3
    )

    print("=" * 60)
    print("Order Book LSTM Model")
    print("=" * 60)
    print(model)
    print(f"\nTotal parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size = 32
    sequence_length = 100
    input_size = 16

    x = torch.randn(batch_size, sequence_length, input_size)

    logits, hidden = model(x)
    probs = model.predict_proba(x)
    predictions = model.predict(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Probabilities shape: {probs.shape}")
    print(f"Predictions shape: {predictions.shape}")

    print(f"\nSample predictions:")
    print(f"  Logits: {logits[0]}")
    print(f"  Probabilities: {probs[0]}")
    print(f"  Predicted class: {predictions[0]}")

    # Test attention LSTM
    print("\n" + "=" * 60)
    print("Attention LSTM Model")
    print("=" * 60)

    attn_model = create_model(
        model_type="attention_lstm",
        input_size=16,
        hidden_size=128,
        num_layers=2,
        attention_heads=4,
    )

    print(f"Total parameters: {count_parameters(attn_model):,}")

    attn_logits = attn_model(x)
    print(f"\nOutput shape: {attn_logits.shape}")
