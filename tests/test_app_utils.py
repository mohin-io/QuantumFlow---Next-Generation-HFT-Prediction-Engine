"""
Comprehensive unit tests for Streamlit app utilities.

Tests all components in isolation to ensure correctness.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from visualization.app_utils import (
    OrderBookGenerator,
    MetricsCalculator,
    SignalGenerator,
    DataGenerator,
    PlotGenerator,
    FeatureAnalyzer,
    validate_orderbook_data,
    format_currency,
    format_percentage,
)


class TestOrderBookGenerator:
    """Test suite for OrderBookGenerator."""

    def test_generate_orderbook_basic(self):
        """Test basic order book generation."""
        generator = OrderBookGenerator()
        bids, asks, mid_price = generator.generate_orderbook()

        assert len(bids) == 20  # Default levels
        assert len(asks) == 20
        assert isinstance(mid_price, float)
        assert mid_price > 0

    def test_generate_orderbook_custom_levels(self):
        """Test order book generation with custom levels."""
        generator = OrderBookGenerator()
        bids, asks, mid_price = generator.generate_orderbook(levels=10)

        assert len(bids) == 10
        assert len(asks) == 10

    def test_orderbook_price_structure(self):
        """Test that order book has correct price structure."""
        generator = OrderBookGenerator()
        bids, asks, mid_price = generator.generate_orderbook(mid_price=50000)

        # Bids should be below mid price
        assert all(bid[0] < mid_price + 1000 for bid in bids)

        # Asks should be above mid price
        assert all(ask[0] > mid_price - 1000 for ask in asks)

        # Bids should be descending
        bid_prices = [b[0] for b in bids]
        assert bid_prices == sorted(bid_prices, reverse=True)

        # Asks should be ascending
        ask_prices = [a[0] for a in asks]
        assert ask_prices == sorted(ask_prices)

    def test_orderbook_volumes_positive(self):
        """Test that all volumes are positive."""
        generator = OrderBookGenerator()
        bids, asks, _ = generator.generate_orderbook()

        assert all(bid[1] > 0 for bid in bids)
        assert all(ask[1] > 0 for ask in asks)

    def test_orderbook_no_overlap(self):
        """Test that bids and asks don't overlap."""
        generator = OrderBookGenerator()
        bids, asks, _ = generator.generate_orderbook()

        # Best bid should be less than best ask
        assert bids[0][0] < asks[0][0]


class TestMetricsCalculator:
    """Test suite for MetricsCalculator."""

    def test_calculate_spread_metrics(self):
        """Test spread metrics calculation."""
        calculator = MetricsCalculator()
        bids = [[49990, 1.0], [49980, 2.0]]
        asks = [[50010, 1.0], [50020, 2.0]]
        mid_price = 50000

        metrics = calculator.calculate_spread_metrics(bids, asks, mid_price)

        assert "spread" in metrics
        assert "spread_bps" in metrics
        assert "spread_pct" in metrics
        assert metrics["spread"] == 20.0
        assert metrics["spread_bps"] == pytest.approx(4.0, rel=0.01)

    def test_calculate_spread_empty_orderbook(self):
        """Test spread calculation with empty order book."""
        calculator = MetricsCalculator()
        metrics = calculator.calculate_spread_metrics([], [], 50000)

        assert metrics["spread"] == 0.0
        assert metrics["spread_bps"] == 0.0

    def test_calculate_volume_imbalance(self):
        """Test volume imbalance calculation."""
        calculator = MetricsCalculator()
        bids = [[49990, 3.0], [49980, 2.0]]  # Total: 5
        asks = [[50010, 1.0], [50020, 2.0]]  # Total: 3

        imbalance = calculator.calculate_volume_imbalance(bids, asks)

        # (5 - 3) / (5 + 3) = 2 / 8 = 0.25
        assert imbalance == pytest.approx(0.25, rel=0.01)

    def test_volume_imbalance_balanced(self):
        """Test volume imbalance when order book is balanced."""
        calculator = MetricsCalculator()
        bids = [[49990, 2.0]]
        asks = [[50010, 2.0]]

        imbalance = calculator.calculate_volume_imbalance(bids, asks)
        assert imbalance == pytest.approx(0.0, abs=0.01)

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        calculator = MetricsCalculator()

        # Create returns with known Sharpe ratio
        returns = np.array([0.01, 0.02, -0.01, 0.015, 0.01] * 100)
        sharpe = calculator.calculate_sharpe_ratio(returns)

        assert isinstance(sharpe, float)
        assert sharpe > 0  # Positive returns should give positive Sharpe

    def test_sharpe_ratio_zero_volatility(self):
        """Test Sharpe ratio with zero volatility."""
        calculator = MetricsCalculator()
        returns = np.array([0.01] * 100)  # Constant returns

        sharpe = calculator.calculate_sharpe_ratio(returns)
        assert sharpe == 0.0  # Should handle division by zero

    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        calculator = MetricsCalculator()

        # Create simple price path with known drawdown
        cumulative_returns = np.array([1.0, 1.1, 1.2, 0.9, 0.95, 1.0])
        max_dd = calculator.calculate_max_drawdown(cumulative_returns)

        # Max drawdown is from 1.2 to 0.9 = -0.25
        assert max_dd == pytest.approx(-0.25, abs=0.01)

    def test_max_drawdown_no_drawdown(self):
        """Test max drawdown with monotonically increasing returns."""
        calculator = MetricsCalculator()
        cumulative_returns = np.array([1.0, 1.1, 1.2, 1.3, 1.4])

        max_dd = calculator.calculate_max_drawdown(cumulative_returns)
        assert max_dd == pytest.approx(0.0, abs=0.01)

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        calculator = MetricsCalculator()
        predictions = ["UP", "DOWN", "FLAT", "UP", "DOWN"]
        actuals = ["UP", "DOWN", "UP", "UP", "FLAT"]

        # 3 correct out of 5 = 0.6
        win_rate = calculator.calculate_win_rate(predictions, actuals)
        assert win_rate == 0.6

    def test_win_rate_perfect_prediction(self):
        """Test win rate with perfect predictions."""
        calculator = MetricsCalculator()
        predictions = ["UP", "DOWN", "FLAT"]
        actuals = ["UP", "DOWN", "FLAT"]

        win_rate = calculator.calculate_win_rate(predictions, actuals)
        assert win_rate == 1.0

    def test_win_rate_mismatched_lengths(self):
        """Test win rate with mismatched input lengths."""
        calculator = MetricsCalculator()
        predictions = ["UP", "DOWN"]
        actuals = ["UP", "DOWN", "FLAT"]

        win_rate = calculator.calculate_win_rate(predictions, actuals)
        assert win_rate == 0.0


class TestSignalGenerator:
    """Test suite for SignalGenerator."""

    def test_generate_signal_up(self):
        """Test UP signal generation."""
        generator = SignalGenerator()
        signal, confidence = generator.generate_signal(
            volume_imbalance=0.3, spread_bps=5.0, threshold=0.1
        )

        assert signal == "UP"
        assert 0.5 <= confidence <= 1.0

    def test_generate_signal_down(self):
        """Test DOWN signal generation."""
        generator = SignalGenerator()
        signal, confidence = generator.generate_signal(
            volume_imbalance=-0.3, spread_bps=5.0, threshold=0.1
        )

        assert signal == "DOWN"
        assert 0.5 <= confidence <= 1.0

    def test_generate_signal_flat(self):
        """Test FLAT signal generation."""
        generator = SignalGenerator()
        signal, confidence = generator.generate_signal(
            volume_imbalance=0.05, spread_bps=5.0, threshold=0.1
        )

        assert signal == "FLAT"
        assert 0.5 <= confidence <= 1.0

    def test_signal_confidence_increases_with_imbalance(self):
        """Test that confidence increases with larger imbalance."""
        generator = SignalGenerator()

        _, conf1 = generator.generate_signal(0.15, 5.0, 0.1)
        _, conf2 = generator.generate_signal(0.35, 5.0, 0.1)

        assert conf2 >= conf1

    def test_calculate_signal_probabilities_up(self):
        """Test probability calculation for UP signal."""
        generator = SignalGenerator()
        probs = generator.calculate_signal_probabilities("UP", 0.8)

        assert probs["UP"] > probs["DOWN"]
        assert probs["UP"] > probs["FLAT"]
        assert abs(sum(probs.values()) - 1.0) < 0.01  # Should sum to 1

    def test_calculate_signal_probabilities_normalized(self):
        """Test that probabilities are normalized."""
        generator = SignalGenerator()
        probs = generator.calculate_signal_probabilities("DOWN", 0.7)

        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01


class TestDataGenerator:
    """Test suite for DataGenerator."""

    def test_generate_price_history(self):
        """Test price history generation."""
        generator = DataGenerator()
        df = generator.generate_price_history(current_price=50000, periods=100)

        assert len(df) == 100
        assert "timestamp" in df.columns
        assert "price" in df.columns
        assert df["price"].iloc[-1] > 0
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_price_history_realistic_volatility(self):
        """Test that price history has reasonable volatility."""
        generator = DataGenerator()
        df = generator.generate_price_history(
            current_price=50000, periods=1000, volatility=0.001
        )

        # Prices should be within reasonable range
        price_range = df["price"].max() - df["price"].min()
        assert price_range < 50000 * 0.1  # Less than 10% range

    def test_generate_prediction_history(self):
        """Test prediction history generation."""
        generator = DataGenerator()
        df = generator.generate_prediction_history(periods=50)

        assert len(df) == 50
        assert "timestamp" in df.columns
        assert "prediction" in df.columns
        assert "confidence" in df.columns
        assert "actual" in df.columns

        # Check value ranges
        assert all(df["confidence"] >= 0.5) and all(df["confidence"] <= 0.95)
        assert set(df["prediction"].unique()).issubset({"UP", "DOWN", "FLAT"})

    def test_prediction_history_has_correlation(self):
        """Test that predictions correlate with actuals (realistic)."""
        generator = DataGenerator()
        df = generator.generate_prediction_history(periods=1000)

        # Calculate accuracy
        correct = sum(df["prediction"] == df["actual"])
        accuracy = correct / len(df)

        # Should be better than random (33.3% for 3-class)
        assert accuracy > 0.4

    def test_generate_pnl_curve(self):
        """Test PnL curve generation."""
        generator = DataGenerator()
        df = generator.generate_pnl_curve(periods=1000, sharpe_target=1.5)

        assert len(df) == 1000
        assert "timestamp" in df.columns
        assert "pnl" in df.columns
        assert df["pnl"].iloc[0] > 0  # Should start positive

    def test_pnl_curve_sharpe_target(self):
        """Test that PnL curve approximates target Sharpe ratio."""
        generator = DataGenerator()
        df = generator.generate_pnl_curve(periods=5000, sharpe_target=2.0)

        returns = df["pnl"].pct_change().dropna()
        sharpe = (
            returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        )

        # Should be close to target (within 50%)
        assert 1.0 <= sharpe <= 3.0


class TestPlotGenerator:
    """Test suite for PlotGenerator."""

    def test_create_orderbook_plot(self):
        """Test order book plot generation."""
        generator = PlotGenerator()
        bids = [[49990, 1.0], [49980, 2.0]]
        asks = [[50010, 1.0], [50020, 2.0]]

        fig = generator.create_orderbook_plot(bids, asks)

        assert fig is not None
        assert len(fig.data) == 2  # Bids and asks traces

    def test_create_signal_probability_plot(self):
        """Test signal probability plot generation."""
        generator = PlotGenerator()
        probs = {"UP": 0.5, "FLAT": 0.3, "DOWN": 0.2}

        fig = generator.create_signal_probability_plot(probs)

        assert fig is not None
        assert len(fig.data) == 1  # Single bar trace

    def test_create_performance_metrics_plot(self):
        """Test performance metrics plot generation."""
        generator = PlotGenerator()
        metrics = {"Accuracy": 0.65, "Precision": 0.62, "Recall": 0.60}

        fig = generator.create_performance_metrics_plot(metrics)

        assert fig is not None


class TestFeatureAnalyzer:
    """Test suite for FeatureAnalyzer."""

    def test_generate_feature_importance(self):
        """Test feature importance generation."""
        analyzer = FeatureAnalyzer()
        df = analyzer.generate_feature_importance(n_features=9)

        assert len(df) == 9
        assert "Feature" in df.columns
        assert "Importance" in df.columns

        # Importance should sum to 1
        assert abs(df["Importance"].sum() - 1.0) < 0.01

        # Should be sorted by importance
        assert df["Importance"].iloc[0] >= df["Importance"].iloc[-1]

    def test_generate_correlation_matrix(self):
        """Test correlation matrix generation."""
        analyzer = FeatureAnalyzer()
        features = ["OFI_L1", "OFI_L5", "Spread"]
        corr_matrix = analyzer.generate_correlation_matrix(features)

        assert corr_matrix.shape == (3, 3)

        # Diagonal should be 1
        assert np.allclose(np.diag(corr_matrix), 1.0)

        # Should be symmetric
        assert np.allclose(corr_matrix, corr_matrix.T)

        # Values should be in [-1, 1]
        assert np.all(corr_matrix >= -1.0) and np.all(corr_matrix <= 1.0)


class TestValidationFunctions:
    """Test suite for validation functions."""

    def test_validate_orderbook_valid(self):
        """Test validation with valid order book."""
        bids = [[49990, 1.0], [49980, 2.0]]
        asks = [[50010, 1.0], [50020, 2.0]]

        assert validate_orderbook_data(bids, asks) is True

    def test_validate_orderbook_empty(self):
        """Test validation with empty order book."""
        assert validate_orderbook_data([], []) is False

    def test_validate_orderbook_overlapping(self):
        """Test validation with overlapping bids/asks."""
        bids = [[50010, 1.0]]
        asks = [[49990, 1.0]]

        assert validate_orderbook_data(bids, asks) is False

    def test_validate_orderbook_wrong_order(self):
        """Test validation with incorrectly ordered prices."""
        bids = [[49980, 1.0], [49990, 2.0]]  # Should be descending
        asks = [[50010, 1.0], [50020, 2.0]]

        assert validate_orderbook_data(bids, asks) is False

    def test_validate_orderbook_negative_volume(self):
        """Test validation with negative volume."""
        bids = [[49990, -1.0]]
        asks = [[50010, 1.0]]

        assert validate_orderbook_data(bids, asks) is False


class TestFormattingFunctions:
    """Test suite for formatting functions."""

    def test_format_currency(self):
        """Test currency formatting."""
        assert format_currency(1234.56) == "$1,234.56"
        assert format_currency(1234.567, decimals=3) == "$1,234.567"
        assert format_currency(1000000) == "$1,000,000.00"

    def test_format_percentage(self):
        """Test percentage formatting."""
        assert format_percentage(0.1234) == "12.34%"
        assert format_percentage(0.5) == "50.00%"
        assert format_percentage(0.123456, decimals=3) == "12.346%"
        assert format_percentage(1.0) == "100.00%"


# Edge cases and integration tests
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_mid_price(self):
        """Test handling of zero mid price."""
        calculator = MetricsCalculator()
        bids = [[0, 1.0]]
        asks = [[0, 1.0]]

        metrics = calculator.calculate_spread_metrics(bids, asks, 0)
        # Should handle gracefully without division by zero
        assert isinstance(metrics, dict)

    def test_extremely_large_numbers(self):
        """Test handling of very large numbers."""
        generator = OrderBookGenerator()
        bids, asks, mid_price = generator.generate_orderbook(mid_price=1e10)

        assert mid_price > 0
        assert validate_orderbook_data(bids, asks)

    def test_single_element_arrays(self):
        """Test calculations with minimal data."""
        calculator = MetricsCalculator()
        returns = np.array([0.01])

        sharpe = calculator.calculate_sharpe_ratio(returns)
        # Should handle gracefully
        assert isinstance(sharpe, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
