"""
Unit tests for feature calculators.

Tests cover:
- Order Flow Imbalance (OFI)
- Micro-price calculation
- Volume profiles
- Realized volatility estimators
"""

import unittest
import numpy as np
import pandas as pd
from dataclasses import dataclass
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.order_flow_imbalance import OFICalculator, OrderBookState
from features.micro_price import MicroPriceCalculator
from features.volume_profile import VolumeProfileCalculator
from features.realized_volatility import RealizedVolatilityEstimator


class TestOFICalculator(unittest.TestCase):
    """Test Order Flow Imbalance calculations."""

    def setUp(self):
        """Set up test fixtures."""
        self.calculator = OFICalculator()

    def test_basic_ofi_positive(self):
        """Test OFI calculation with increased bid volume."""
        # Previous state
        prev_state = OrderBookState(
            timestamp=1000,
            bids=[(100.0, 10.0), (99.9, 15.0)],
            asks=[(100.1, 12.0), (100.2, 18.0)]
        )

        # Current state - bid volume increased
        curr_state = OrderBookState(
            timestamp=1001,
            bids=[(100.0, 20.0), (99.9, 15.0)],  # +10 at best bid
            asks=[(100.1, 12.0), (100.2, 18.0)]
        )

        self.calculator.update(prev_state)
        ofi_metrics = self.calculator.compute_ofi(curr_state, num_levels=2)

        # OFI should be positive (bid volume increased)
        self.assertGreater(ofi_metrics['ofi_level_1'], 0)

    def test_basic_ofi_negative(self):
        """Test OFI calculation with increased ask volume."""
        prev_state = OrderBookState(
            timestamp=1000,
            bids=[(100.0, 10.0), (99.9, 15.0)],
            asks=[(100.1, 12.0), (100.2, 18.0)]
        )

        # Ask volume increased
        curr_state = OrderBookState(
            timestamp=1001,
            bids=[(100.0, 10.0), (99.9, 15.0)],
            asks=[(100.1, 25.0), (100.2, 18.0)]  # +13 at best ask
        )

        self.calculator.update(prev_state)
        ofi_metrics = self.calculator.compute_ofi(curr_state, num_levels=2)

        # OFI should be negative (ask volume increased)
        self.assertLess(ofi_metrics['ofi_level_1'], 0)

    def test_ofi_no_change(self):
        """Test OFI when order book unchanged."""
        state = OrderBookState(
            timestamp=1000,
            bids=[(100.0, 10.0), (99.9, 15.0)],
            asks=[(100.1, 12.0), (100.2, 18.0)]
        )

        self.calculator.update(state)
        ofi_metrics = self.calculator.compute_ofi(state, num_levels=2)

        # OFI should be zero
        self.assertEqual(ofi_metrics['ofi_level_1'], 0.0)


class TestMicroPriceCalculator(unittest.TestCase):
    """Test micro-price calculations."""

    def setUp(self):
        """Set up test fixtures."""
        self.calculator = MicroPriceCalculator()

    def test_micro_price_basic(self):
        """Test basic micro-price calculation."""
        state = OrderBookState(
            timestamp=1000,
            bids=[(100.0, 10.0)],
            asks=[(100.2, 10.0)]
        )

        micro_price = self.calculator.calculate(state)

        # Micro-price should be between bid and ask
        self.assertGreater(micro_price, 100.0)
        self.assertLess(micro_price, 100.2)

    def test_micro_price_weighted(self):
        """Test micro-price is volume-weighted."""
        # More volume on bid side
        state1 = OrderBookState(
            timestamp=1000,
            bids=[(100.0, 100.0)],  # Large bid volume
            asks=[(100.2, 10.0)]
        )

        # More volume on ask side
        state2 = OrderBookState(
            timestamp=1001,
            bids=[(100.0, 10.0)],
            asks=[(100.2, 100.0)]  # Large ask volume
        )

        price1 = self.calculator.calculate(state1)
        price2 = self.calculator.calculate(state2)

        # Price1 should be closer to bid (higher bid volume)
        # Price2 should be closer to ask (higher ask volume)
        mid = 100.1
        self.assertLess(abs(price1 - 100.0), abs(price1 - 100.2))
        self.assertLess(abs(price2 - 100.2), abs(price2 - 100.0))

    def test_spread(self):
        """Test bid-ask spread calculation."""
        state = OrderBookState(
            timestamp=1000,
            bids=[(100.0, 10.0)],
            asks=[(100.5, 10.0)]
        )

        spread = self.calculator.calculate_spread(state)
        self.assertEqual(spread, 0.5)

        spread_bps = self.calculator.calculate_spread_bps(state)
        self.assertAlmostEqual(spread_bps, 50.0, places=1)  # 0.5/100 * 10000


class TestVolumeProfileCalculator(unittest.TestCase):
    """Test volume profile calculations."""

    def setUp(self):
        """Set up test fixtures."""
        self.calculator = VolumeProfileCalculator()

    def test_volume_imbalance(self):
        """Test volume imbalance calculation."""
        # Bid-heavy order book
        state = OrderBookState(
            timestamp=1000,
            bids=[(100.0, 50.0), (99.9, 30.0)],
            asks=[(100.1, 20.0), (100.2, 10.0)]
        )

        metrics = self.calculator.calculate(state, num_levels=2)

        # Should be positive (more bid volume)
        self.assertGreater(metrics['volume_imbalance_2'], 0)

    def test_volume_depth_ratio(self):
        """Test volume depth ratio."""
        state = OrderBookState(
            timestamp=1000,
            bids=[(100.0, 40.0), (99.9, 30.0), (99.8, 20.0)],
            asks=[(100.1, 30.0), (100.2, 20.0), (100.3, 10.0)]
        )

        metrics = self.calculator.calculate(state, num_levels=3)

        # Depth ratio should be > 1 (more bid depth)
        self.assertGreater(metrics['depth_ratio_3'], 1.0)


class TestRealizedVolatilityEstimator(unittest.TestCase):
    """Test realized volatility estimators."""

    def test_simple_rv(self):
        """Test simple realized volatility."""
        # Generate returns
        prices = pd.Series([100, 101, 99, 102, 98, 103])
        returns = prices.pct_change().dropna()

        rv = RealizedVolatilityEstimator.simple_rv(returns, window=5)

        # Should be positive
        self.assertGreater(rv.iloc[-1], 0)

    def test_parkinson_volatility(self):
        """Test Parkinson volatility estimator."""
        # Generate high/low prices
        high = pd.Series([102, 103, 101, 104, 100, 105])
        low = pd.Series([98, 99, 97, 100, 96, 101])

        vol = RealizedVolatilityEstimator.parkinson_volatility(high, low, window=5)

        # Should be positive
        self.assertGreater(vol.iloc[-1], 0)

    def test_garman_klass_volatility(self):
        """Test Garman-Klass volatility estimator."""
        open_prices = pd.Series([100, 101, 99, 102, 98, 103])
        high = pd.Series([102, 103, 101, 104, 100, 105])
        low = pd.Series([98, 99, 97, 100, 96, 101])
        close = pd.Series([101, 100, 102, 99, 103, 98])

        vol = RealizedVolatilityEstimator.garman_klass_volatility(
            open_prices, high, low, close, window=5
        )

        # Should be positive
        self.assertGreater(vol.iloc[-1], 0)

    def test_volatility_consistency(self):
        """Test that different estimators give consistent results."""
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5))
        high = prices + np.abs(np.random.randn(100) * 0.3)
        low = prices - np.abs(np.random.randn(100) * 0.3)
        returns = prices.pct_change().dropna()

        simple_vol = RealizedVolatilityEstimator.simple_rv(returns, window=20).iloc[-1]
        park_vol = RealizedVolatilityEstimator.parkinson_volatility(high, low, window=20).iloc[-1]

        # Both should be positive and of similar magnitude
        self.assertGreater(simple_vol, 0)
        self.assertGreater(park_vol, 0)
        # Order of magnitude check (within 10x)
        self.assertLess(abs(simple_vol - park_vol) / simple_vol, 10.0)


class TestFeaturePipeline(unittest.TestCase):
    """Integration tests for complete feature pipeline."""

    def test_feature_pipeline_integration(self):
        """Test that all features can be computed together."""
        from features.feature_pipeline import FeaturePipeline, FeaturePipelineConfig

        # Generate synthetic order book snapshots
        snapshots = []
        for i in range(50):
            state = OrderBookState(
                timestamp=1000 + i,
                bids=[
                    (100.0 - 0.01 * j, 10.0 + np.random.rand() * 5)
                    for j in range(10)
                ],
                asks=[
                    (100.0 + 0.01 * (j + 1), 10.0 + np.random.rand() * 5)
                    for j in range(10)
                ]
            )
            snapshots.append(state)

        # Create pipeline
        config = FeaturePipelineConfig(
            ofi_levels=[1, 5],
            ofi_windows=[10, 20],
            volatility_windows=[10, 20]
        )
        pipeline = FeaturePipeline(config)

        # Compute features
        features_df = pipeline.compute_all_features(snapshots)

        # Verify features exist
        self.assertIsNotNone(features_df)
        self.assertGreater(len(features_df), 0)
        self.assertGreater(len(features_df.columns), 10)

        # Check for NaN handling
        self.assertLess(features_df.isnull().sum().sum() / features_df.size, 0.5)


if __name__ == '__main__':
    print("="*80)
    print("RUNNING FEATURE CALCULATOR UNIT TESTS")
    print("="*80)

    # Run tests with verbosity
    unittest.main(verbosity=2)
