"""
Micro-price and Fair Value Calculations

Micro-price is a volume-weighted average of the best bid and ask prices,
providing a more accurate estimate of the "true" price than the mid-price.

Mathematical Definition:
P_micro = (V_ask × P_bid + V_bid × P_ask) / (V_bid + V_ask)

Where:
- P_bid: Best bid price
- P_ask: Best ask price
- V_bid: Volume at best bid
- V_ask: Volume at best ask

The micro-price is superior to mid-price because it accounts for volume imbalances
and provides better predictions of the next trade price.

References:
- Stoikov, S. (2018). The micro-price: a high-frequency estimator of future prices
- Cartea, Á., Jaimungal, S., & Penalva, J. (2015). Algorithmic and High-Frequency Trading
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class MicroPriceMetrics:
    """Container for micro-price and related metrics."""

    micro_price: float
    mid_price: float
    weighted_mid_price: float
    micro_price_deviation: float  # Deviation from mid-price
    micro_price_bps_deviation: float  # Deviation in basis points


class MicroPriceCalculator:
    """
    Calculator for micro-price and fair value estimates.

    Provides multiple methods for estimating the fair value of an asset
    based on order book information.
    """

    @staticmethod
    def compute_micro_price(
        bid_price: float,
        ask_price: float,
        bid_volume: float,
        ask_volume: float
    ) -> float:
        """
        Compute volume-weighted micro-price.

        Args:
            bid_price: Best bid price
            ask_price: Best ask price
            bid_volume: Volume at best bid
            ask_volume: Volume at best ask

        Returns:
            Micro-price
        """
        total_volume = bid_volume + ask_volume

        if total_volume == 0:
            return (bid_price + ask_price) / 2

        micro_price = (ask_volume * bid_price + bid_volume * ask_price) / total_volume
        return micro_price

    @staticmethod
    def compute_weighted_mid_price(
        bid_price: float,
        ask_price: float,
        bid_volume: float,
        ask_volume: float,
        depth_levels: int = 3
    ) -> float:
        """
        Compute volume-weighted mid-price using multiple depth levels.

        Args:
            bid_price: Best bid price (or list of bid prices)
            ask_price: Best ask price (or list of ask prices)
            bid_volume: Bid volume (or list of volumes)
            ask_volume: Ask volume (or list of volumes)
            depth_levels: Number of levels to consider

        Returns:
            Weighted mid-price
        """
        # Handle single level
        if isinstance(bid_price, (int, float)):
            return MicroPriceCalculator.compute_micro_price(
                bid_price, ask_price, bid_volume, ask_volume
            )

        # Multiple levels
        total_bid_volume = sum(bid_volume[:depth_levels])
        total_ask_volume = sum(ask_volume[:depth_levels])

        if total_bid_volume + total_ask_volume == 0:
            return (bid_price[0] + ask_price[0]) / 2

        weighted_bid = sum(p * v for p, v in zip(bid_price[:depth_levels], bid_volume[:depth_levels]))
        weighted_ask = sum(p * v for p, v in zip(ask_price[:depth_levels], ask_volume[:depth_levels]))

        weighted_mid = (weighted_bid + weighted_ask) / (total_bid_volume + total_ask_volume)
        return weighted_mid

    @staticmethod
    def compute_vwap_price(
        bid_prices: List[float],
        ask_prices: List[float],
        bid_volumes: List[float],
        ask_volumes: List[float],
        depth_levels: int = 5
    ) -> float:
        """
        Compute Volume-Weighted Average Price (VWAP) from order book.

        Args:
            bid_prices: List of bid prices
            ask_prices: List of ask prices
            bid_volumes: List of bid volumes
            ask_volumes: List of ask volumes
            depth_levels: Number of levels to consider

        Returns:
            VWAP
        """
        all_prices = []
        all_volumes = []

        for i in range(min(depth_levels, len(bid_prices), len(ask_prices))):
            all_prices.extend([bid_prices[i], ask_prices[i]])
            all_volumes.extend([bid_volumes[i], ask_volumes[i]])

        total_volume = sum(all_volumes)

        if total_volume == 0:
            return (bid_prices[0] + ask_prices[0]) / 2

        vwap = sum(p * v for p, v in zip(all_prices, all_volumes)) / total_volume
        return vwap

    @staticmethod
    def compute_all_metrics(
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        depth_levels: int = 3
    ) -> MicroPriceMetrics:
        """
        Compute all micro-price related metrics.

        Args:
            bids: List of (price, volume) tuples for bids
            asks: List of (price, volume) tuples for asks
            depth_levels: Number of levels for weighted calculations

        Returns:
            MicroPriceMetrics object
        """
        if not bids or not asks:
            return MicroPriceMetrics(0, 0, 0, 0, 0)

        bid_price, bid_volume = bids[0]
        ask_price, ask_volume = asks[0]

        # Micro-price (top level only)
        micro_price = MicroPriceCalculator.compute_micro_price(
            bid_price, ask_price, bid_volume, ask_volume
        )

        # Mid-price
        mid_price = (bid_price + ask_price) / 2

        # Weighted mid-price (multiple levels)
        bid_prices = [b[0] for b in bids[:depth_levels]]
        ask_prices = [a[0] for a in asks[:depth_levels]]
        bid_volumes = [b[1] for b in bids[:depth_levels]]
        ask_volumes = [a[1] for a in asks[:depth_levels]]

        weighted_mid = MicroPriceCalculator.compute_weighted_mid_price(
            bid_prices, ask_prices, bid_volumes, ask_volumes, depth_levels
        )

        # Deviation metrics
        deviation = micro_price - mid_price
        deviation_bps = (deviation / mid_price * 10000) if mid_price > 0 else 0

        return MicroPriceMetrics(
            micro_price=micro_price,
            mid_price=mid_price,
            weighted_mid_price=weighted_mid,
            micro_price_deviation=deviation,
            micro_price_bps_deviation=deviation_bps
        )


class AdaptiveFairValueEstimator:
    """
    Adaptive fair value estimator with exponential smoothing.

    Combines micro-price with historical data using exponential weighting.
    """

    def __init__(self, alpha: float = 0.3):
        """
        Initialize adaptive estimator.

        Args:
            alpha: Smoothing factor (0 < alpha < 1)
                  Higher alpha = more weight on recent observations
        """
        self.alpha = alpha
        self.fair_value: Optional[float] = None

    def update(self, micro_price: float) -> float:
        """
        Update fair value estimate with new micro-price observation.

        Args:
            micro_price: Latest micro-price

        Returns:
            Updated fair value estimate
        """
        if self.fair_value is None:
            self.fair_value = micro_price
        else:
            self.fair_value = self.alpha * micro_price + (1 - self.alpha) * self.fair_value

        return self.fair_value

    def reset(self):
        """Reset the estimator."""
        self.fair_value = None


def compute_micro_price_features(
    df: pd.DataFrame,
    depth_levels: int = 3,
    use_adaptive: bool = True,
    alpha: float = 0.3
) -> pd.DataFrame:
    """
    Compute micro-price features from DataFrame of order book snapshots.

    Args:
        df: DataFrame with 'bids' and 'asks' columns
        depth_levels: Number of levels for weighted calculations
        use_adaptive: Whether to compute adaptive fair value
        alpha: Smoothing factor for adaptive estimator

    Returns:
        DataFrame with micro-price features added
    """
    features = []
    adaptive_estimator = AdaptiveFairValueEstimator(alpha=alpha) if use_adaptive else None

    for idx, row in df.iterrows():
        bids = row['bids'] if isinstance(row['bids'], list) else []
        asks = row['asks'] if isinstance(row['asks'], list) else []

        # Convert to tuples if needed
        if bids and not isinstance(bids[0], tuple):
            bids = [tuple(b) for b in bids]
        if asks and not isinstance(asks[0], tuple):
            asks = [tuple(a) for a in asks]

        # Compute metrics
        metrics = MicroPriceCalculator.compute_all_metrics(bids, asks, depth_levels)

        feature_dict = {
            'micro_price': metrics.micro_price,
            'mid_price': metrics.mid_price,
            'weighted_mid_price': metrics.weighted_mid_price,
            'micro_price_deviation': metrics.micro_price_deviation,
            'micro_price_bps_deviation': metrics.micro_price_bps_deviation,
        }

        # Add adaptive fair value
        if use_adaptive and metrics.micro_price > 0:
            fair_value = adaptive_estimator.update(metrics.micro_price)
            feature_dict['adaptive_fair_value'] = fair_value
            feature_dict['price_vs_fair_value'] = metrics.micro_price - fair_value
            feature_dict['price_vs_fair_value_bps'] = (
                (metrics.micro_price - fair_value) / fair_value * 10000
                if fair_value > 0 else 0
            )

        features.append(feature_dict)

    # Convert to DataFrame and merge
    features_df = pd.DataFrame(features)
    result_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)

    return result_df


# Example usage and testing
if __name__ == "__main__":
    # Example: Generate synthetic order book data
    np.random.seed(42)

    snapshots = []
    mid_price = 100.0

    for i in range(500):
        mid_price += np.random.normal(0, 0.02)

        bids = []
        asks = []

        for level in range(10):
            bid_price = mid_price - 0.01 * (level + 1)
            ask_price = mid_price + 0.01 * (level + 1)

            # Add volume imbalance to make micro-price interesting
            imbalance = np.random.normal(0, 0.2)
            bid_volume = max(10, 50 + 20 * imbalance + np.random.uniform(-10, 10))
            ask_volume = max(10, 50 - 20 * imbalance + np.random.uniform(-10, 10))

            bids.append([bid_price, bid_volume])
            asks.append([ask_price, ask_volume])

        snapshots.append({
            'timestamp': i,
            'bids': bids,
            'asks': asks
        })

    df = pd.DataFrame(snapshots)

    # Compute micro-price features
    print("Computing micro-price features...")
    df_with_features = compute_micro_price_features(df, depth_levels=3, use_adaptive=True)

    print("\n" + "="*80)
    print("Micro-Price Features (first 10 rows)")
    print("="*80)

    feature_cols = ['micro_price', 'mid_price', 'weighted_mid_price',
                    'micro_price_deviation', 'adaptive_fair_value']
    print(df_with_features[['timestamp'] + feature_cols].head(10))

    print("\n" + "="*80)
    print("Feature Statistics")
    print("="*80)
    print(df_with_features[feature_cols].describe())

    # Analyze correlation between micro-price deviation and future price changes
    df_with_features['future_mid_price'] = df_with_features['mid_price'].shift(-10)
    df_with_features['future_return'] = (
        (df_with_features['future_mid_price'] - df_with_features['mid_price']) /
        df_with_features['mid_price']
    )

    correlation = df_with_features[['micro_price_bps_deviation', 'future_return']].corr()
    print("\n" + "="*80)
    print("Predictive Power Analysis")
    print("="*80)
    print("Correlation between micro-price deviation and future returns:")
    print(correlation)

    # Visualization
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # Plot 1: Micro-price vs Mid-price
        axes[0].plot(df_with_features['timestamp'], df_with_features['mid_price'],
                     label='Mid Price', alpha=0.7, linewidth=1)
        axes[0].plot(df_with_features['timestamp'], df_with_features['micro_price'],
                     label='Micro Price', alpha=0.7, linewidth=1)
        axes[0].plot(df_with_features['timestamp'], df_with_features['adaptive_fair_value'],
                     label='Adaptive Fair Value', alpha=0.7, linewidth=1, linestyle='--')
        axes[0].set_title('Price Comparison: Mid-Price vs Micro-Price vs Fair Value')
        axes[0].set_xlabel('Tick')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Plot 2: Micro-price deviation
        axes[1].plot(df_with_features['timestamp'], df_with_features['micro_price_bps_deviation'],
                     label='Micro-Price Deviation (bps)', alpha=0.7, color='orange')
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.3)
        axes[1].fill_between(df_with_features['timestamp'],
                             df_with_features['micro_price_bps_deviation'],
                             0, alpha=0.3)
        axes[1].set_title('Micro-Price Deviation from Mid-Price')
        axes[1].set_xlabel('Tick')
        axes[1].set_ylabel('Deviation (bps)')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('data/simulations/micro_price_example.png', dpi=150)
        print("\nVisualization saved to: data/simulations/micro_price_example.png")

    except ImportError:
        print("\nMatplotlib not available for visualization")
