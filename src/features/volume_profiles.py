"""
Volume Profiles and Liquidity Metrics

Analyzes the distribution of liquidity across price levels in the order book.

Key Metrics:
- Total bid/ask volume
- Volume imbalance
- Depth imbalance
- Liquidity concentration
- Volume-weighted spread
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class VolumeProfileMetrics:
    """Container for volume profile metrics."""

    total_bid_volume: float
    total_ask_volume: float
    volume_imbalance: float
    volume_imbalance_ratio: float

    depth_imbalance: float
    liquidity_concentration_bid: float
    liquidity_concentration_ask: float

    vwap_bid: float
    vwap_ask: float
    volume_weighted_spread: float


class VolumeProfileCalculator:
    """Calculator for volume profile and liquidity metrics."""

    @staticmethod
    def compute_volume_metrics(
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        depth_levels: int = 20,
    ) -> VolumeProfileMetrics:
        """
        Compute comprehensive volume profile metrics.

        Args:
            bids: List of (price, volume) tuples
            asks: List of (price, volume) tuples
            depth_levels: Number of levels to analyze

        Returns:
            VolumeProfileMetrics object
        """
        if not bids or not asks:
            return VolumeProfileMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        # Extract prices and volumes
        bid_prices = np.array([b[0] for b in bids[:depth_levels]])
        bid_volumes = np.array([b[1] for b in bids[:depth_levels]])
        ask_prices = np.array([a[0] for a in asks[:depth_levels]])
        ask_volumes = np.array([a[1] for a in asks[:depth_levels]])

        # Total volumes
        total_bid_vol = np.sum(bid_volumes)
        total_ask_vol = np.sum(ask_volumes)

        # Volume imbalance
        total_vol = total_bid_vol + total_ask_vol
        vol_imbalance = total_bid_vol - total_ask_vol
        vol_imbalance_ratio = vol_imbalance / total_vol if total_vol > 0 else 0

        # Depth imbalance (ratio of bid to ask volume)
        depth_imbalance = total_bid_vol / total_ask_vol if total_ask_vol > 0 else 0

        # Liquidity concentration (what % of volume is in top 3 levels)
        top_n = min(3, len(bid_volumes), len(ask_volumes))
        liq_conc_bid = (
            np.sum(bid_volumes[:top_n]) / total_bid_vol if total_bid_vol > 0 else 0
        )
        liq_conc_ask = (
            np.sum(ask_volumes[:top_n]) / total_ask_vol if total_ask_vol > 0 else 0
        )

        # Volume-weighted average prices
        vwap_bid = (
            np.sum(bid_prices * bid_volumes) / total_bid_vol
            if total_bid_vol > 0
            else bid_prices[0]
        )
        vwap_ask = (
            np.sum(ask_prices * ask_volumes) / total_ask_vol
            if total_ask_vol > 0
            else ask_prices[0]
        )

        # Volume-weighted spread
        vw_spread = vwap_ask - vwap_bid

        return VolumeProfileMetrics(
            total_bid_volume=total_bid_vol,
            total_ask_volume=total_ask_vol,
            volume_imbalance=vol_imbalance,
            volume_imbalance_ratio=vol_imbalance_ratio,
            depth_imbalance=depth_imbalance,
            liquidity_concentration_bid=liq_conc_bid,
            liquidity_concentration_ask=liq_conc_ask,
            vwap_bid=vwap_bid,
            vwap_ask=vwap_ask,
            volume_weighted_spread=vw_spread,
        )

    @staticmethod
    def compute_spread_metrics(
        bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]
    ) -> Dict[str, float]:
        """Compute various spread metrics."""
        if not bids or not asks:
            return {"spread": 0, "spread_bps": 0, "relative_spread": 0}

        bid_price = bids[0][0]
        ask_price = asks[0][0]
        mid_price = (bid_price + ask_price) / 2

        spread = ask_price - bid_price
        spread_bps = (spread / mid_price * 10000) if mid_price > 0 else 0
        relative_spread = spread / mid_price if mid_price > 0 else 0

        return {
            "spread": spread,
            "spread_bps": spread_bps,
            "relative_spread": relative_spread,
        }


def compute_volume_features(df: pd.DataFrame, depth_levels: int = 20) -> pd.DataFrame:
    """
    Compute volume profile features from DataFrame.

    Args:
        df: DataFrame with 'bids' and 'asks' columns
        depth_levels: Number of levels to analyze

    Returns:
        DataFrame with volume features added
    """
    features = []

    for idx, row in df.iterrows():
        bids = row["bids"] if isinstance(row["bids"], list) else []
        asks = row["asks"] if isinstance(row["asks"], list) else []

        # Convert to tuples if needed
        if bids and not isinstance(bids[0], tuple):
            bids = [tuple(b) for b in bids]
        if asks and not isinstance(asks[0], tuple):
            asks = [tuple(a) for a in asks]

        # Compute metrics
        metrics = VolumeProfileCalculator.compute_volume_metrics(
            bids, asks, depth_levels
        )
        spread_metrics = VolumeProfileCalculator.compute_spread_metrics(bids, asks)

        feature_dict = {
            "total_bid_volume": metrics.total_bid_volume,
            "total_ask_volume": metrics.total_ask_volume,
            "volume_imbalance": metrics.volume_imbalance,
            "volume_imbalance_ratio": metrics.volume_imbalance_ratio,
            "depth_imbalance": metrics.depth_imbalance,
            "liquidity_concentration_bid": metrics.liquidity_concentration_bid,
            "liquidity_concentration_ask": metrics.liquidity_concentration_ask,
            **spread_metrics,
        }

        features.append(feature_dict)

    features_df = pd.DataFrame(features)
    result_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)

    return result_df


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)

    snapshots = []
    for i in range(100):
        mid_price = 100 + np.random.normal(0, 1)

        bids = [
            [mid_price - 0.01 * (j + 1), 50 + np.random.uniform(-20, 20)]
            for j in range(20)
        ]
        asks = [
            [mid_price + 0.01 * (j + 1), 50 + np.random.uniform(-20, 20)]
            for j in range(20)
        ]

        snapshots.append({"timestamp": i, "bids": bids, "asks": asks})

    df = pd.DataFrame(snapshots)
    df_with_features = compute_volume_features(df)

    print(
        df_with_features[
            ["volume_imbalance_ratio", "depth_imbalance", "spread_bps"]
        ].head(10)
    )
