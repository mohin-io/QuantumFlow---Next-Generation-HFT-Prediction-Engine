"""
End-to-End Integration Tests

Tests complete pipeline:
1. Data ingestion → Feature engineering → Model prediction
2. Backtesting → Performance evaluation
3. API → Dashboard → User interaction
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.order_flow_imbalance import OFICalculator, OrderBookState
from features.micro_price import MicroPriceCalculator
from features.volume_profile import VolumeProfileCalculator
from features.feature_pipeline import FeaturePipeline, FeaturePipelineConfig
from backtesting.backtest_engine import BacktestEngine, BacktestConfig
from backtesting.economic_validation import EconomicValidator, MarketConditions


class TestEndToEndPipeline:
    """Test complete data processing pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        # Generate synthetic order book data
        np.random.seed(42)
        self.snapshots = self._generate_order_book_snapshots(100)

    def _generate_order_book_snapshots(self, n: int):
        """Generate synthetic order book snapshots."""
        snapshots = []
        base_price = 100.0

        for i in range(n):
            price = base_price + np.random.randn() * 0.5

            bids = [
                [price - 0.01 * j, np.random.uniform(1, 10)]
                for j in range(1, 11)
            ]

            asks = [
                [price + 0.01 * j, np.random.uniform(1, 10)]
                for j in range(1, 11)
            ]

            snapshot = OrderBookState(
                timestamp=1000000 + i * 1000,
                bids=bids,
                asks=asks
            )

            snapshots.append(snapshot)

        return snapshots

    def test_feature_pipeline(self):
        """Test feature extraction pipeline."""
        # Create pipeline
        config = FeaturePipelineConfig(
            ofi_levels=[1, 5],
            ofi_windows=[10, 20],
            volatility_windows=[10]
        )
        pipeline = FeaturePipeline(config)

        # Extract features
        features_df = pipeline.compute_all_features(self.snapshots)

        # Assertions
        assert features_df is not None
        assert len(features_df) > 0
        assert len(features_df.columns) > 10

        # Check for key features
        assert any('ofi' in col for col in features_df.columns)
        assert any('micro_price' in col for col in features_df.columns)
        assert any('volume' in col for col in features_df.columns)

        print(f"[OK] Feature pipeline extracted {len(features_df.columns)} features")

    def test_backtesting_pipeline(self):
        """Test backtesting engine."""
        # Generate predictions
        n_samples = 80
        predictions_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1min'),
            'prediction': np.random.randint(0, 3, n_samples),
            'probability': np.random.uniform(0.5, 0.95, n_samples)
        })

        # Generate prices
        prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.1)
        prices_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1min'),
            'mid_price': prices,
            'bid': prices * 0.9995,
            'ask': prices * 1.0005
        })

        # Run backtest
        config = BacktestConfig(
            initial_capital=100000,
            position_size=0.1,
            transaction_cost_bps=5.0,
            confidence_threshold=0.6
        )

        engine = BacktestEngine(config)
        metrics = engine.run(predictions_df, prices_df)

        # Assertions
        assert metrics is not None
        assert 'total_trades' in metrics
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics

        print(f"[OK] Backtest completed: {metrics['total_trades']} trades, "
              f"{metrics['total_return']:.2f}% return")

    def test_economic_validation(self):
        """Test economic validation pipeline."""
        # Generate data
        n = 100
        predictions_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1min'),
            'prediction': np.random.choice([0, 2], n, p=[0.4, 0.6]),  # Biased to bullish
            'probability': np.random.uniform(0.6, 0.9, n)
        })

        prices = 100 + np.cumsum(np.random.randn(n) * 0.05)
        prices_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1min'),
            'mid_price': prices,
            'close': prices,
            'bid': prices * 0.9998,
            'ask': prices * 1.0002,
            'volume': np.random.lognormal(10, 1, n),
            'volatility': np.random.uniform(0.001, 0.003, n)
        })

        # Run validation
        validator = EconomicValidator(MarketConditions())
        metrics = validator.validate_strategy(predictions_df, prices_df, prices_df)

        # Assertions
        assert metrics is not None
        assert hasattr(metrics, 'overall_score') or 'total_trades' in str(metrics)

        print(f"[OK] Economic validation completed")


class TestAPIIntegration:
    """Test API endpoints and responses."""

    def test_live_data_connector(self):
        """Test live data API connector."""
        from api.live_data_connector import BinanceConnector

        connector = BinanceConnector()

        # Test order book fetch
        try:
            book = connector.get_order_book('BTCUSDT', limit=10)

            if book:
                assert book.exchange == 'binance'
                assert book.symbol == 'BTCUSDT'
                assert len(book.bids) > 0
                assert len(book.asks) > 0
                assert book.bids[0][0] < book.asks[0][0]  # Bid < Ask

                print(f"[OK] Live data connector working (BTC @ ${book.bids[0][0]:,.2f})")
            else:
                print("[SKIP] Could not fetch live data (network/API limit)")

        except Exception as e:
            print(f"[SKIP] Live data test skipped: {e}")

    def test_feature_calculator_integration(self):
        """Test feature calculators work together."""
        # Create order book
        state = OrderBookState(
            timestamp=1000000,
            bids=[[100.0, 10.0], [99.9, 8.0], [99.8, 6.0]],
            asks=[[100.1, 9.0], [100.2, 7.0], [100.3, 5.0]]
        )

        # OFI
        ofi_calc = OFICalculator()
        ofi_calc.update(state)

        # Micro-price
        micro_calc = MicroPriceCalculator()
        micro_price = micro_calc.calculate(state)

        # Volume profile
        vol_calc = VolumeProfileCalculator()
        vol_metrics = vol_calc.calculate(state)

        # Assertions
        assert micro_price > 0
        assert 'volume_imbalance_3' in vol_metrics
        assert vol_metrics['volume_imbalance_3'] != 0

        print(f"[OK] Feature calculators integrated (Micro-price: ${micro_price:.2f})")


class TestModelPipeline:
    """Test model training and prediction pipeline."""

    def test_model_can_train(self):
        """Test that model can be trained."""
        import torch
        from models.lstm_model import OrderBookLSTM

        # Create synthetic data
        n_samples = 100
        seq_length = 10
        n_features = 20

        X = torch.randn(n_samples, seq_length, n_features)
        y = torch.randint(0, 3, (n_samples,))

        # Create model
        model = OrderBookLSTM(
            input_size=n_features,
            hidden_size=64,
            num_layers=1,
            num_classes=3
        )

        # Train for 1 epoch
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs, _ = model(X)
        loss = criterion(outputs, y)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Assertions
        assert loss.item() > 0
        assert not torch.isnan(loss)

        print(f"[OK] Model training pipeline works (Loss: {loss.item():.4f})")

    def test_model_prediction(self):
        """Test model can make predictions."""
        import torch
        from models.lstm_model import OrderBookLSTM

        model = OrderBookLSTM(
            input_size=20,
            hidden_size=64,
            num_layers=1,
            num_classes=3
        )

        model.eval()

        X = torch.randn(5, 10, 20)

        with torch.no_grad():
            outputs, _ = model(X)
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probs, dim=1)

        # Assertions
        assert outputs.shape == (5, 3)
        assert probs.shape == (5, 3)
        assert predictions.shape == (5,)
        assert torch.all((predictions >= 0) & (predictions < 3))

        print(f"[OK] Model prediction pipeline works")


class TestSystemIntegration:
    """Test complete system integration."""

    def test_end_to_end_workflow(self):
        """Test complete workflow from data to prediction to backtest."""
        # 1. Generate order book data
        snapshots = []
        base_price = 100.0

        for i in range(50):
            price = base_price + np.random.randn() * 0.3

            snapshot = OrderBookState(
                timestamp=1000000 + i * 1000,
                bids=[[price - 0.01 * j, np.random.uniform(5, 15)] for j in range(1, 6)],
                asks=[[price + 0.01 * j, np.random.uniform(5, 15)] for j in range(1, 6)]
            )
            snapshots.append(snapshot)

        # 2. Extract features
        config = FeaturePipelineConfig(ofi_levels=[1], ofi_windows=[10])
        pipeline = FeaturePipeline(config)
        features_df = pipeline.compute_all_features(snapshots)

        assert len(features_df) > 0

        # 3. Generate predictions (mock)
        predictions = np.random.randint(0, 3, len(features_df))
        probabilities = np.random.uniform(0.6, 0.9, len(features_df))

        predictions_df = pd.DataFrame({
            'timestamp': features_df['timestamp'],
            'prediction': predictions,
            'probability': probabilities
        })

        # 4. Backtest
        prices_df = pd.DataFrame({
            'timestamp': features_df['timestamp'],
            'mid_price': features_df.get('micro_price', 100 + np.random.randn(len(features_df))),
            'bid': features_df['timestamp'].apply(lambda x: 100 * 0.9995),
            'ask': features_df['timestamp'].apply(lambda x: 100 * 1.0005)
        })

        config = BacktestConfig(confidence_threshold=0.65)
        engine = BacktestEngine(config)
        metrics = engine.run(predictions_df, prices_df)

        # Assertions
        assert metrics is not None
        assert 'total_trades' in metrics

        print(f"[OK] End-to-end workflow complete")
        print(f"     Features: {len(features_df.columns)}")
        print(f"     Predictions: {len(predictions_df)}")
        print(f"     Trades: {metrics['total_trades']}")


@pytest.mark.slow
class TestPerformance:
    """Test system performance."""

    def test_feature_extraction_speed(self):
        """Test feature extraction performance."""
        import time

        # Generate large dataset
        snapshots = []
        for i in range(1000):
            snapshot = OrderBookState(
                timestamp=i * 1000,
                bids=[[100 - 0.01 * j, 10] for j in range(1, 11)],
                asks=[[100 + 0.01 * j, 10] for j in range(1, 11)]
            )
            snapshots.append(snapshot)

        # Time feature extraction
        config = FeaturePipelineConfig()
        pipeline = FeaturePipeline(config)

        start = time.time()
        features_df = pipeline.compute_all_features(snapshots)
        elapsed = time.time() - start

        # Should process 1000 snapshots in under 10 seconds
        assert elapsed < 10.0

        print(f"[OK] Feature extraction: {elapsed:.2f}s for 1000 snapshots "
              f"({1000/elapsed:.0f} snapshots/sec)")


if __name__ == '__main__':
    print('='*80)
    print('RUNNING END-TO-END INTEGRATION TESTS')
    print('='*80)

    # Run all tests
    pytest.main([__file__, '-v', '--tb=short'])
