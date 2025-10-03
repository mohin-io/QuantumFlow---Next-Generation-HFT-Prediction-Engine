"""
Generate Sample Plots for Documentation

This script generates all the visualizations referenced in the notebooks
and README for demonstration purposes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append('..')

# Create output directory
output_dir = Path('../data/simulations')
output_dir.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("GENERATING SAMPLE VISUALIZATIONS")
print("="*80)

# Generate synthetic data
np.random.seed(42)
n_samples = 1000

# 1. Order Book Heatmap Example
print("\n1. Generating order book heatmap...")
mid_price = 50000
bids = [[mid_price - i * 0.5, 50 * np.exp(-i * 0.1) + np.random.uniform(-5, 5)] for i in range(1, 21)]
asks = [[mid_price + i * 0.5, 50 * np.exp(-i * 0.1) + np.random.uniform(-5, 5)] for i in range(1, 21)]

fig, ax = plt.subplots(figsize=(12, 8))
bid_prices = [b[0] for b in bids]
bid_vols = [b[1] for b in bids]
ask_prices = [a[0] for a in asks]
ask_vols = [a[1] for a in asks]

ax.barh(bid_prices, bid_vols, color='green', alpha=0.7, label='Bids')
ax.barh(ask_prices, [-v for v in ask_vols], color='red', alpha=0.7, label='Asks')
ax.axhline(y=mid_price, color='black', linestyle='--', linewidth=2, label=f'Mid Price: ${mid_price:,}')
ax.set_xlabel('Volume', fontsize=12)
ax.set_ylabel('Price ($)', fontsize=12)
ax.set_title('Order Book Depth Visualization - BTCUSDT', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(output_dir / 'order_book_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Saved: order_book_heatmap.png")

# 2. OFI Time Series
print("\n2. Generating OFI time series...")
timestamps = np.arange(n_samples) * 0.1
ofi_l1 = np.cumsum(np.random.normal(0, 5, n_samples))
ofi_l5 = np.cumsum(np.random.normal(0, 8, n_samples))
ofi_l10 = np.cumsum(np.random.normal(0, 10, n_samples))

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

axes[0].plot(timestamps, ofi_l1, alpha=0.8, linewidth=1.5, color='blue')
axes[0].axhline(0, color='red', linestyle='--', alpha=0.5)
axes[0].fill_between(timestamps, ofi_l1, 0, alpha=0.3, color='blue')
axes[0].set_title('Order Flow Imbalance - Level 1 (Best Bid/Ask)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('OFI')
axes[0].grid(alpha=0.3)

axes[1].plot(timestamps, ofi_l5, alpha=0.8, linewidth=1.5, color='orange')
axes[1].axhline(0, color='red', linestyle='--', alpha=0.5)
axes[1].fill_between(timestamps, ofi_l5, 0, alpha=0.3, color='orange')
axes[1].set_title('Order Flow Imbalance - Top 5 Levels', fontsize=12, fontweight='bold')
axes[1].set_ylabel('OFI')
axes[1].grid(alpha=0.3)

axes[2].plot(timestamps, ofi_l10, alpha=0.8, linewidth=1.5, color='green')
axes[2].axhline(0, color='red', linestyle='--', alpha=0.5)
axes[2].fill_between(timestamps, ofi_l10, 0, alpha=0.3, color='green')
axes[2].set_title('Order Flow Imbalance - Top 10 Levels', fontsize=12, fontweight='bold')
axes[2].set_ylabel('OFI')
axes[2].set_xlabel('Time (seconds)')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'ofi_multi_level.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Saved: ofi_multi_level.png")

# 3. Feature Correlation Heatmap
print("\n3. Generating feature correlation heatmap...")
feature_names = [
    'OFI_L1', 'OFI_L5', 'OFI_L10',
    'Micro_Price_Dev', 'Volume_Imb',
    'Depth_Imb', 'Spread_BPS',
    'Liquidity_Conc', 'Cancel_Ratio'
]

# Generate synthetic correlation matrix
n_features = len(feature_names)
corr_matrix = np.random.uniform(-0.5, 0.5, (n_features, n_features))
np.fill_diagonal(corr_matrix, 1.0)
# Make symmetric
corr_matrix = (corr_matrix + corr_matrix.T) / 2

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=1,
            xticklabels=feature_names, yticklabels=feature_names,
            cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'feature_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Saved: feature_correlation.png")

# 4. Training Curves
print("\n4. Generating model training curves...")
epochs = np.arange(1, 31)
train_loss = 1.2 * np.exp(-epochs * 0.08) + 0.4 + np.random.normal(0, 0.02, len(epochs))
val_loss = 1.2 * np.exp(-epochs * 0.07) + 0.5 + np.random.normal(0, 0.03, len(epochs))
train_acc = 1 - np.exp(-epochs * 0.1) * 0.6 + np.random.normal(0, 0.01, len(epochs))
val_acc = 1 - np.exp(-epochs * 0.09) * 0.65 + np.random.normal(0, 0.015, len(epochs))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epochs, train_loss, linewidth=2, marker='o', markersize=4, label='Train Loss')
axes[0].plot(epochs, val_loss, linewidth=2, marker='s', markersize=4, label='Val Loss')
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('Loss', fontsize=11)
axes[0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

axes[1].plot(epochs, train_acc, linewidth=2, marker='o', markersize=4, label='Train Accuracy')
axes[1].plot(epochs, val_acc, linewidth=2, marker='s', markersize=4, label='Val Accuracy')
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('Accuracy', fontsize=11)
axes[1].set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Saved: training_curves.png")

# 5. Confusion Matrix
print("\n5. Generating confusion matrix...")
cm = np.array([[45, 8, 12], [10, 52, 18], [15, 20, 55]])
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Down', 'Flat', 'Up'],
            yticklabels=['Down', 'Flat', 'Up'],
            ax=axes[0])
axes[0].set_title('Confusion Matrix (Counts)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=['Down', 'Flat', 'Up'],
            yticklabels=['Down', 'Flat', 'Up'],
            ax=axes[1])
axes[1].set_title('Confusion Matrix (Normalized)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Saved: confusion_matrix.png")

# 6. Price and Spread Evolution
print("\n6. Generating price evolution plot...")
timestamps = np.arange(500) * 0.1
price = 50000 + np.cumsum(np.random.normal(0, 5, 500))
spread_bps = 2.0 + np.random.gamma(2, 0.5, 500)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(timestamps, price, linewidth=1.5, alpha=0.8)
axes[0].set_ylabel('Price ($)', fontsize=11)
axes[0].set_title('Mid-Price Evolution - BTCUSDT', fontsize=12, fontweight='bold')
axes[0].grid(alpha=0.3)

axes[1].plot(timestamps, spread_bps, linewidth=1.5, alpha=0.8, color='orange')
axes[1].set_xlabel('Time (seconds)', fontsize=11)
axes[1].set_ylabel('Spread (bps)', fontsize=11)
axes[1].set_title('Bid-Ask Spread Over Time', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'price_spread_evolution.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Saved: price_spread_evolution.png")

# 7. Volume Imbalance
print("\n7. Generating volume imbalance plot...")
volume_imbalance = np.random.normal(0, 0.2, 500)
volume_imbalance = pd.Series(volume_imbalance).rolling(20).mean().fillna(0).values

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(timestamps, volume_imbalance, linewidth=1.5, alpha=0.8, color='purple')
ax.axhline(0, color='red', linestyle='--', alpha=0.5)
ax.fill_between(timestamps, volume_imbalance, 0, alpha=0.3, color='purple')
ax.set_xlabel('Time (seconds)', fontsize=11)
ax.set_ylabel('Volume Imbalance Ratio', fontsize=11)
ax.set_title('Order Book Volume Imbalance (Bid - Ask) / Total', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'volume_imbalance.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Saved: volume_imbalance.png")

# 8. Architecture Diagram (simple text-based for now)
print("\n8. Generating architecture overview...")
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

architecture_text = """
HFT ORDER BOOK FORECASTING SYSTEM ARCHITECTURE

┌─────────────────────────────────────────────────────────┐
│              DATA INGESTION LAYER                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │ Binance  │  │ Coinbase │  │ LOBSTER  │             │
│  │ WebSocket│  │ WebSocket│  │  Data    │             │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘             │
│       └────────────┬─────────────┘                     │
└────────────────────┼───────────────────────────────────┘
                     │
┌────────────────────▼───────────────────────────────────┐
│              KAFKA STREAMING                           │
└────────────────────┬───────────────────────────────────┘
                     │
┌────────────────────▼───────────────────────────────────┐
│         POSTGRESQL + TIMESCALEDB                       │
└────────────────────┬───────────────────────────────────┘
                     │
┌────────────────────▼───────────────────────────────────┐
│         FEATURE ENGINEERING (60+ features)             │
│  • Order Flow Imbalance (OFI)                          │
│  • Micro-price & Fair Value                            │
│  • Volume Profiles & Liquidity                         │
│  • Queue Dynamics                                      │
│  • Realized Volatility                                 │
└────────────────────┬───────────────────────────────────┘
                     │
┌────────────────────▼───────────────────────────────────┐
│              ML MODELS                                 │
│  LSTM (2 layers, 128 hidden) → Attention → Ensemble   │
└────────────────────┬───────────────────────────────────┘
                     │
┌────────────────────▼───────────────────────────────────┐
│         FASTAPI SERVICE + STREAMLIT DASHBOARD          │
│         (Real-time predictions <50ms)                  │
└─────────────────────────────────────────────────────────┘
"""

ax.text(0.5, 0.5, architecture_text, fontsize=9, family='monospace',
        ha='center', va='center', transform=ax.transAxes)
ax.set_title('System Architecture Overview', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(output_dir / 'system_architecture.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Saved: system_architecture.png")

print("\n" + "="*80)
print("VISUALIZATION GENERATION COMPLETE")
print("="*80)
print(f"\nAll plots saved to: {output_dir}")
print("\nGenerated files:")
print("  1. order_book_heatmap.png")
print("  2. ofi_multi_level.png")
print("  3. feature_correlation.png")
print("  4. training_curves.png")
print("  5. confusion_matrix.png")
print("  6. price_spread_evolution.png")
print("  7. volume_imbalance.png")
print("  8. system_architecture.png")
print("\n[OK] Ready to embed in README and documentation!")
print("="*80)
