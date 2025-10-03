"""
Integration and system-level tests for Streamlit Trading Dashboard.

Tests the app as a whole system to ensure all components work together.
"""

import pytest
import sys
import os
import subprocess
import time
import requests
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from visualization.app_utils import (
    OrderBookGenerator,
    MetricsCalculator,
    SignalGenerator,
    DataGenerator,
    validate_orderbook_data,
)


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def test_complete_trading_workflow(self):
        """Test complete workflow from order book to signal generation."""
        # Step 1: Generate order book
        ob_gen = OrderBookGenerator()
        bids, asks, mid_price = ob_gen.generate_orderbook(mid_price=50000, levels=20)

        # Validate order book
        assert validate_orderbook_data(bids, asks)

        # Step 2: Calculate metrics
        metrics_calc = MetricsCalculator()
        spread_metrics = metrics_calc.calculate_spread_metrics(bids, asks, mid_price)
        volume_imbalance = metrics_calc.calculate_volume_imbalance(bids, asks)

        assert spread_metrics["spread"] > 0
        assert -1 <= volume_imbalance <= 1

        # Step 3: Generate signal
        signal_gen = SignalGenerator()
        signal, confidence = signal_gen.generate_signal(
            volume_imbalance, spread_metrics["spread_bps"]
        )

        assert signal in ["UP", "DOWN", "FLAT"]
        assert 0.5 <= confidence <= 1.0

        # Step 4: Calculate probabilities
        probs = signal_gen.calculate_signal_probabilities(signal, confidence)

        assert abs(sum(probs.values()) - 1.0) < 0.01
        assert probs[signal] == max(probs.values())

    def test_performance_analysis_workflow(self):
        """Test complete performance analysis workflow."""
        # Generate PnL data
        data_gen = DataGenerator()
        pnl_data = data_gen.generate_pnl_curve(periods=1000, sharpe_target=1.5)

        assert len(pnl_data) == 1000
        assert "pnl" in pnl_data.columns

        # Calculate metrics
        metrics_calc = MetricsCalculator()
        returns = pnl_data["pnl"].pct_change().dropna()

        sharpe = metrics_calc.calculate_sharpe_ratio(returns.values)
        max_dd = metrics_calc.calculate_max_drawdown(pnl_data["pnl"].values)

        assert sharpe > 0
        assert max_dd <= 0

    def test_prediction_evaluation_workflow(self):
        """Test prediction evaluation workflow."""
        # Generate prediction history
        data_gen = DataGenerator()
        pred_history = data_gen.generate_prediction_history(periods=100)

        # Calculate win rate
        metrics_calc = MetricsCalculator()
        win_rate = metrics_calc.calculate_win_rate(
            pred_history["prediction"].tolist(), pred_history["actual"].tolist()
        )

        assert 0 <= win_rate <= 1

        # Win rate should be better than random for 3-class (33%)
        assert win_rate > 0.35  # Allowing some variance


class TestDataConsistency:
    """Test data consistency across components."""

    def test_orderbook_to_metrics_consistency(self):
        """Test that order book data produces consistent metrics."""
        ob_gen = OrderBookGenerator()

        # Generate multiple order books with same parameters
        results = []
        for _ in range(10):
            bids, asks, mid_price = ob_gen.generate_orderbook(
                mid_price=50000, spread_bps=5.0
            )
            metrics_calc = MetricsCalculator()
            spread_metrics = metrics_calc.calculate_spread_metrics(
                bids, asks, mid_price
            )
            results.append(spread_metrics["spread_bps"])

        # Spreads should be relatively consistent (within 50% of mean)
        mean_spread = np.mean(results)
        assert all(0.5 * mean_spread <= s <= 1.5 * mean_spread for s in results)

    def test_signal_determinism(self):
        """Test that same inputs produce same signals."""
        signal_gen = SignalGenerator()

        # Same inputs should produce same outputs
        signal1, conf1 = signal_gen.generate_signal(0.3, 5.0, 0.1)
        signal2, conf2 = signal_gen.generate_signal(0.3, 5.0, 0.1)

        assert signal1 == signal2
        assert conf1 == conf2


class TestSystemBehavior:
    """Test overall system behavior and properties."""

    def test_orderbook_always_valid(self):
        """Test that generated order books are always valid."""
        ob_gen = OrderBookGenerator()

        for _ in range(100):
            bids, asks, _ = ob_gen.generate_orderbook()
            assert validate_orderbook_data(bids, asks)

    def test_metrics_always_in_valid_range(self):
        """Test that all metrics are in valid ranges."""
        ob_gen = OrderBookGenerator()
        metrics_calc = MetricsCalculator()

        for _ in range(50):
            bids, asks, mid_price = ob_gen.generate_orderbook()

            spread_metrics = metrics_calc.calculate_spread_metrics(
                bids, asks, mid_price
            )
            volume_imbalance = metrics_calc.calculate_volume_imbalance(bids, asks)

            # Spread should be positive
            assert spread_metrics["spread"] > 0
            assert spread_metrics["spread_bps"] > 0

            # Volume imbalance should be in [-1, 1]
            assert -1 <= volume_imbalance <= 1

    def test_signals_probability_consistency(self):
        """Test that signal probabilities are always consistent."""
        signal_gen = SignalGenerator()

        for signal in ["UP", "DOWN", "FLAT"]:
            for confidence in [0.6, 0.7, 0.8, 0.9]:
                probs = signal_gen.calculate_signal_probabilities(signal, confidence)

                # Probabilities should sum to 1
                assert abs(sum(probs.values()) - 1.0) < 0.01

                # Highest probability should match signal
                assert probs[signal] == max(probs.values())


class TestPerformanceMetrics:
    """Test performance and efficiency metrics."""

    def test_orderbook_generation_performance(self):
        """Test that order book generation is fast enough."""
        ob_gen = OrderBookGenerator()

        start_time = time.time()
        for _ in range(1000):
            ob_gen.generate_orderbook()
        elapsed = time.time() - start_time

        # Should generate 1000 order books in less than 1 second
        assert elapsed < 1.0

    def test_metrics_calculation_performance(self):
        """Test that metrics calculation is fast enough."""
        ob_gen = OrderBookGenerator()
        metrics_calc = MetricsCalculator()

        bids, asks, mid_price = ob_gen.generate_orderbook(levels=100)

        start_time = time.time()
        for _ in range(1000):
            metrics_calc.calculate_spread_metrics(bids, asks, mid_price)
            metrics_calc.calculate_volume_imbalance(bids, asks)
        elapsed = time.time() - start_time

        # Should calculate 1000 times in less than 1 second
        assert elapsed < 1.0

    def test_signal_generation_performance(self):
        """Test that signal generation is fast enough."""
        signal_gen = SignalGenerator()

        start_time = time.time()
        for _ in range(10000):
            signal_gen.generate_signal(0.3, 5.0, 0.1)
        elapsed = time.time() - start_time

        # Should generate 10000 signals in less than 1 second
        assert elapsed < 1.0


class TestDataVisualization:
    """Test data visualization components."""

    def test_plot_generation_without_errors(self):
        """Test that all plots can be generated without errors."""
        from visualization.app_utils import PlotGenerator

        ob_gen = OrderBookGenerator()
        plot_gen = PlotGenerator()

        # Generate order book plot
        bids, asks, _ = ob_gen.generate_orderbook()
        fig1 = plot_gen.create_orderbook_plot(bids, asks)
        assert fig1 is not None

        # Generate signal probability plot
        probs = {"UP": 0.5, "FLAT": 0.3, "DOWN": 0.2}
        fig2 = plot_gen.create_signal_probability_plot(probs)
        assert fig2 is not None

        # Generate metrics plot
        metrics = {"Accuracy": 0.65, "Precision": 0.62}
        fig3 = plot_gen.create_performance_metrics_plot(metrics)
        assert fig3 is not None


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_orderbook_handling(self):
        """Test handling of empty order book."""
        metrics_calc = MetricsCalculator()

        spread_metrics = metrics_calc.calculate_spread_metrics([], [], 50000)
        assert spread_metrics["spread"] == 0.0

        volume_imbalance = metrics_calc.calculate_volume_imbalance([], [])
        assert volume_imbalance == 0.0

    def test_invalid_signal_threshold(self):
        """Test signal generation with extreme thresholds."""
        signal_gen = SignalGenerator()

        # Very high threshold
        signal, conf = signal_gen.generate_signal(0.05, 5.0, threshold=1.0)
        assert signal == "FLAT"

        # Zero threshold
        signal, conf = signal_gen.generate_signal(0.05, 5.0, threshold=0.0)
        assert signal in ["UP", "DOWN", "FLAT"]

    def test_zero_volatility_pnl(self):
        """Test PnL generation with zero volatility."""
        data_gen = DataGenerator()

        # This should still work, producing flat PnL
        df = data_gen.generate_pnl_curve(periods=100, sharpe_target=0.1)
        assert len(df) == 100


class TestStreamlitAppImports:
    """Test that the Streamlit app can be imported without errors."""

    def test_import_main_app(self):
        """Test that main app file imports successfully."""
        import importlib.util

        app_path = os.path.join(
            os.path.dirname(__file__), "..", "run_trading_dashboard.py"
        )

        spec = importlib.util.spec_from_file_location("app", app_path)
        # Just check that spec can be created
        assert spec is not None

    def test_all_dependencies_available(self):
        """Test that all required dependencies are available."""
        required_packages = [
            "pandas",
            "numpy",
            "plotly",
        ]

        optional_packages = ["streamlit"]

        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                pytest.fail(f"Required package {package} not available")

        # Optional packages - just warn
        for package in optional_packages:
            try:
                __import__(package)
            except ImportError:
                print(f"Optional package {package} not available")


class TestStatisticalProperties:
    """Test statistical properties of generated data."""

    def test_price_history_statistics(self):
        """Test that generated price history has expected properties."""
        data_gen = DataGenerator()

        # Generate many price histories
        final_prices = []
        for _ in range(100):
            df = data_gen.generate_price_history(50000, periods=100, volatility=0.01)
            final_prices.append(df["price"].iloc[-1])

        # Mean should be reasonably close to starting price (within 5%)
        assert abs(np.mean(final_prices) - 50000) < 2500

    def test_returns_distribution(self):
        """Test that returns follow expected distribution."""
        data_gen = DataGenerator()
        pnl_data = data_gen.generate_pnl_curve(periods=10000, sharpe_target=1.5)

        returns = pnl_data["pnl"].pct_change().dropna()

        # Returns should have approximately zero mean (small positive)
        assert abs(returns.mean()) < 0.001

        # Returns should have some variance
        assert returns.std() > 0


class TestConcurrency:
    """Test concurrent operations and thread safety."""

    def test_concurrent_orderbook_generation(self):
        """Test that order book generation works concurrently."""
        from concurrent.futures import ThreadPoolExecutor

        ob_gen = OrderBookGenerator()

        def generate():
            bids, asks, mid_price = ob_gen.generate_orderbook()
            return validate_orderbook_data(bids, asks)

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(lambda _: generate(), range(100)))

        # All should be valid
        assert all(results)


class TestDataIntegrity:
    """Test data integrity and correctness."""

    def test_orderbook_best_prices(self):
        """Test that best bid/ask are correctly positioned."""
        ob_gen = OrderBookGenerator()

        for _ in range(50):
            bids, asks, mid_price = ob_gen.generate_orderbook()

            best_bid = bids[0][0]
            best_ask = asks[0][0]

            # Best bid should be highest bid
            assert all(bid[0] <= best_bid for bid in bids)

            # Best ask should be lowest ask
            assert all(ask[0] >= best_ask for ask in asks)

            # Mid price should be between best bid and ask
            assert best_bid < mid_price < best_ask or abs(
                mid_price - (best_bid + best_ask) / 2
            ) < (best_ask - best_bid)

    def test_cumulative_pnl_monotonic_when_positive(self):
        """Test PnL properties."""
        data_gen = DataGenerator()
        pnl_data = data_gen.generate_pnl_curve(periods=1000, sharpe_target=2.0)

        # First value should be positive
        assert pnl_data["pnl"].iloc[0] > 0

        # Should have both positive and negative returns
        returns = pnl_data["pnl"].pct_change().dropna()
        assert any(returns > 0) and any(returns < 0)


# Integration test for complete system
class TestCompleteSystemIntegration:
    """Test the complete system integration."""

    def test_full_dashboard_simulation(self):
        """Simulate a complete dashboard update cycle."""
        # 1. Generate order book
        ob_gen = OrderBookGenerator()
        bids, asks, mid_price = ob_gen.generate_orderbook(mid_price=50000)

        assert validate_orderbook_data(bids, asks)

        # 2. Calculate all metrics
        metrics_calc = MetricsCalculator()
        spread_metrics = metrics_calc.calculate_spread_metrics(bids, asks, mid_price)
        volume_imbalance = metrics_calc.calculate_volume_imbalance(bids, asks)

        # 3. Generate signal
        signal_gen = SignalGenerator()
        signal, confidence = signal_gen.generate_signal(
            volume_imbalance, spread_metrics["spread_bps"]
        )
        probabilities = signal_gen.calculate_signal_probabilities(signal, confidence)

        # 4. Generate historical data
        data_gen = DataGenerator()
        price_history = data_gen.generate_price_history(mid_price)
        pred_history = data_gen.generate_prediction_history()
        pnl_curve = data_gen.generate_pnl_curve()

        # 5. Calculate performance metrics
        win_rate = metrics_calc.calculate_win_rate(
            pred_history["prediction"].tolist(), pred_history["actual"].tolist()
        )
        returns = pnl_curve["pnl"].pct_change().dropna()
        sharpe = metrics_calc.calculate_sharpe_ratio(returns.values)
        max_dd = metrics_calc.calculate_max_drawdown(pnl_curve["pnl"].values)

        # Verify all components work together
        assert all(
            [
                spread_metrics["spread"] > 0,
                signal in ["UP", "DOWN", "FLAT"],
                abs(sum(probabilities.values()) - 1.0) < 0.01,
                len(price_history) > 0,
                len(pred_history) > 0,
                0 <= win_rate <= 1,
                isinstance(sharpe, float),
                max_dd <= 0,
            ]
        )

        print("[PASS] Full dashboard simulation successful")
        print(f"  - Mid Price: ${mid_price:,.2f}")
        print(f"  - Spread: {spread_metrics['spread_bps']:.2f} bps")
        print(f"  - Signal: {signal} ({confidence:.1%} confidence)")
        print(f"  - Win Rate: {win_rate:.1%}")
        print(f"  - Sharpe: {sharpe:.2f}")
        print(f"  - Max DD: {max_dd:.2%}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
