"""
Transformer Model for Order Book Forecasting

Implements Transformer architecture with:
- Multi-head self-attention
- Positional encoding
- Feed-forward networks
- Better for capturing long-range dependencies than LSTM

References:
- Vaswani et al. (2017). Attention is All You Need
- Zhang et al. (2019). Transformer-based models for limit order book forecasting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer.

    Adds position information to sequence embeddings using sine/cosine functions.
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class OrderBookTransformer(nn.Module):
    """
    Transformer model for order book imbalance prediction.

    Uses encoder-only architecture (similar to BERT) for classification.
    """

    def __init__(
        self,
        input_size,
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        num_classes=3
    ):
        """
        Initialize Transformer model.

        Args:
            input_size: Number of input features
            d_model: Embedding dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            num_classes: Number of output classes
        """
        super(OrderBookTransformer, self).__init__()

        self.input_size = input_size
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, dim_feedforward // 2)
        self.fc2 = nn.Linear(dim_feedforward // 2, num_classes)

        # Layer norm
        self.layer_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            mask: Optional attention mask

        Returns:
            Logits of shape (batch, num_classes)
        """
        # Project input to d_model dimension
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        # Global average pooling over sequence
        x = torch.mean(x, dim=1)  # (batch, d_model)

        # Layer norm
        x = self.layer_norm(x)

        # Classification head
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits

    def get_attention_weights(self, x):
        """
        Extract attention weights for visualization.

        Returns attention weights from all layers.
        """
        # Project input
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        # Store attention weights
        attention_weights = []

        # Need to modify forward pass to extract attention
        # For now, return None (would require custom transformer implementation)
        return None

    def predict_proba(self, x):
        """Predict class probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs

    def predict(self, x):
        """Predict class labels."""
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)


class MultiScaleTransformer(nn.Module):
    """
    Multi-scale Transformer for order book forecasting.

    Processes sequences at different time scales and combines them.
    """

    def __init__(
        self,
        input_size,
        d_model=128,
        nhead=8,
        num_encoder_layers=3,
        dropout=0.1,
        num_classes=3,
        scales=[1, 5, 10]  # Different downsampling factors
    ):
        super(MultiScaleTransformer, self).__init__()

        self.scales = scales

        # Create transformer for each scale
        self.transformers = nn.ModuleList([
            OrderBookTransformer(
                input_size=input_size,
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                dropout=dropout,
                num_classes=d_model  # Output features, not classes
            )
            for _ in scales
        ])

        # Fusion layer
        self.fusion = nn.Linear(d_model * len(scales), d_model)

        # Final classification
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def downsample_sequence(self, x, factor):
        """Downsample sequence by averaging."""
        if factor == 1:
            return x

        batch_size, seq_len, features = x.shape
        new_len = seq_len // factor

        # Reshape and average
        x_trimmed = x[:, :new_len * factor, :]
        x_reshaped = x_trimmed.reshape(batch_size, new_len, factor, features)
        x_downsampled = torch.mean(x_reshaped, dim=2)

        return x_downsampled

    def forward(self, x):
        """
        Forward pass with multi-scale processing.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Logits of shape (batch, num_classes)
        """
        scale_features = []

        for scale, transformer in zip(self.scales, self.transformers):
            # Downsample
            x_scaled = self.downsample_sequence(x, scale)

            # Process with transformer
            # Modify to output features instead of classes
            features = transformer.input_proj(x_scaled)
            features = transformer.pos_encoder(features)
            features = transformer.transformer_encoder(features)
            features = torch.mean(features, dim=1)  # Global pooling

            scale_features.append(features)

        # Concatenate multi-scale features
        combined = torch.cat(scale_features, dim=1)

        # Fusion
        fused = F.relu(self.fusion(combined))

        # Classification
        fused = self.dropout(fused)
        logits = self.fc(fused)

        return logits


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Example usage
if __name__ == "__main__":
    # Test Transformer model
    batch_size = 32
    seq_len = 100
    input_size = 16
    num_classes = 3

    # Create model
    model = OrderBookTransformer(
        input_size=input_size,
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
        dropout=0.1,
        num_classes=num_classes
    )

    print("="*80)
    print("Order Book Transformer Model")
    print("="*80)
    print(model)
    print(f"\nTotal parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(batch_size, seq_len, input_size)

    logits = model(x)
    probs = model.predict_proba(x)
    predictions = model.predict(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Probabilities shape: {probs.shape}")
    print(f"Predictions shape: {predictions.shape}")

    print("\nSample predictions:")
    print(f"  Logits: {logits[0]}")
    print(f"  Probabilities: {probs[0]}")
    print(f"  Predicted class: {predictions[0]}")

    # Test multi-scale transformer
    print("\n" + "="*80)
    print("Multi-Scale Transformer Model")
    print("="*80)

    multi_model = MultiScaleTransformer(
        input_size=input_size,
        d_model=128,
        nhead=8,
        num_encoder_layers=3,
        num_classes=num_classes,
        scales=[1, 5, 10]
    )

    print(f"Total parameters: {count_parameters(multi_model):,}")

    multi_logits = multi_model(x)
    print(f"\nOutput shape: {multi_logits.shape}")
