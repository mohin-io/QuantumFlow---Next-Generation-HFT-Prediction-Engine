"""
Performance Optimization and Profiling Scripts

Includes:
1. Feature calculation optimization
2. Model inference optimization
3. Database query optimization
4. Caching strategies
5. Memory profiling
"""

import numpy as np
import pandas as pd
import time
from functools import wraps, lru_cache
from typing import Callable, Any
import cProfile
import pstats
import io
from memory_profiler import profile as memory_profile
import numba
from numba import jit


def time_it(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[TIME] {func.__name__}: {elapsed*1000:.2f}ms")
        return result
    return wrapper


def profile_it(func: Callable) -> Callable:
    """Decorator to profile function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()

        result = func(*args, **kwargs)

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # Top 10 functions
        print(s.getvalue())

        return result
    return wrapper


# Optimized Feature Calculations with Numba

@jit(nopython=True)
def calculate_ofi_numba(prev_bids, prev_asks, curr_bids, curr_asks, num_levels=10):
    """
    Optimized OFI calculation using Numba JIT compilation.

    Args:
        prev_bids: Previous bid prices and sizes (N x 2 array)
        prev_asks: Previous ask prices and sizes (N x 2 array)
        curr_bids: Current bid prices and sizes (N x 2 array)
        curr_asks: Current ask prices and sizes (N x 2 array)
        num_levels: Number of levels to calculate

    Returns:
        OFI value
    """
    ofi = 0.0

    for level in range(min(num_levels, len(prev_bids), len(curr_bids))):
        # Bid side
        if level < len(prev_bids) and level < len(curr_bids):
            prev_bid_vol = prev_bids[level, 1]
            curr_bid_vol = curr_bids[level, 1]
            delta_bid = curr_bid_vol - prev_bid_vol

            if delta_bid > 0:
                ofi += delta_bid

        # Ask side
        if level < len(prev_asks) and level < len(curr_asks):
            prev_ask_vol = prev_asks[level, 1]
            curr_ask_vol = curr_asks[level, 1]
            delta_ask = curr_ask_vol - prev_ask_vol

            if delta_ask > 0:
                ofi -= delta_ask

    return ofi


@jit(nopython=True)
def calculate_micro_price_numba(bids, asks):
    """
    Optimized micro-price calculation.

    Args:
        bids: Bid prices and sizes (N x 2 array)
        asks: Ask prices and sizes (N x 2 array)

    Returns:
        Micro-price
    """
    if len(bids) == 0 or len(asks) == 0:
        return 0.0

    best_bid_price = bids[0, 0]
    best_bid_vol = bids[0, 1]
    best_ask_price = asks[0, 0]
    best_ask_vol = asks[0, 1]

    if best_bid_vol + best_ask_vol == 0:
        return (best_bid_price + best_ask_price) / 2.0

    micro_price = (best_ask_vol * best_bid_price + best_bid_vol * best_ask_price) / (best_bid_vol + best_ask_vol)

    return micro_price


@jit(nopython=True)
def calculate_volume_imbalance_numba(bids, asks, num_levels=10):
    """
    Optimized volume imbalance calculation.

    Args:
        bids: Bid prices and sizes
        asks: Ask prices and sizes
        num_levels: Number of levels

    Returns:
        Volume imbalance
    """
    total_bid_vol = 0.0
    total_ask_vol = 0.0

    for i in range(min(num_levels, len(bids))):
        total_bid_vol += bids[i, 1]

    for i in range(min(num_levels, len(asks))):
        total_ask_vol += asks[i, 1]

    if total_bid_vol + total_ask_vol == 0:
        return 0.0

    return (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)


class OptimizedFeatureCalculator:
    """
    Optimized feature calculator using:
    - Numba JIT compilation
    - Vectorized operations
    - Caching
    """

    def __init__(self):
        self.cache = {}

    @lru_cache(maxsize=1000)
    def _cached_calc(self, key: str, data_hash: int) -> float:
        """Cached calculation."""
        return self.cache.get(key, 0.0)

    @time_it
    def calculate_features_batch(self, order_books: list) -> pd.DataFrame:
        """
        Calculate features for batch of order books efficiently.

        Args:
            order_books: List of (bids, asks) tuples

        Returns:
            DataFrame with features
        """
        n = len(order_books)

        # Pre-allocate arrays
        ofi_values = np.zeros(n)
        micro_prices = np.zeros(n)
        vol_imbalances = np.zeros(n)
        spreads = np.zeros(n)

        # Process in batch
        for i in range(1, n):
            prev_bids, prev_asks = order_books[i-1]
            curr_bids, curr_asks = order_books[i]

            # Convert to numpy arrays if not already
            prev_bids = np.array(prev_bids) if not isinstance(prev_bids, np.ndarray) else prev_bids
            prev_asks = np.array(prev_asks) if not isinstance(prev_asks, np.ndarray) else prev_asks
            curr_bids = np.array(curr_bids) if not isinstance(curr_bids, np.ndarray) else curr_bids
            curr_asks = np.array(curr_asks) if not isinstance(curr_asks, np.ndarray) else curr_asks

            # Use optimized functions
            ofi_values[i] = calculate_ofi_numba(prev_bids, prev_asks, curr_bids, curr_asks)
            micro_prices[i] = calculate_micro_price_numba(curr_bids, curr_asks)
            vol_imbalances[i] = calculate_volume_imbalance_numba(curr_bids, curr_asks)

            if len(curr_bids) > 0 and len(curr_asks) > 0:
                spreads[i] = curr_asks[0, 0] - curr_bids[0, 0]

        # Create DataFrame
        df = pd.DataFrame({
            'ofi': ofi_values,
            'micro_price': micro_prices,
            'volume_imbalance': vol_imbalances,
            'spread': spreads
        })

        return df


class DatabaseOptimizer:
    """Database query optimization strategies."""

    @staticmethod
    def create_indexes_sql():
        """SQL commands to create optimal indexes."""
        return """
        -- Order book snapshots index
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orderbook_timestamp
        ON orderbook_snapshots(timestamp DESC);

        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orderbook_symbol_timestamp
        ON orderbook_snapshots(symbol, timestamp DESC);

        -- Features index
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_features_timestamp
        ON features(timestamp DESC);

        -- Predictions index
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_predictions_timestamp
        ON predictions(timestamp DESC);

        -- Composite index for common queries
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orderbook_composite
        ON orderbook_snapshots(symbol, exchange, timestamp DESC)
        INCLUDE (bids, asks);

        -- TimescaleDB hypertables
        SELECT create_hypertable('orderbook_snapshots', 'timestamp',
                                 chunk_time_interval => INTERVAL '1 day',
                                 if_not_exists => TRUE);

        SELECT create_hypertable('features', 'timestamp',
                                 chunk_time_interval => INTERVAL '1 day',
                                 if_not_exists => TRUE);

        -- Compression policy (after 7 days)
        SELECT add_compression_policy('orderbook_snapshots', INTERVAL '7 days');
        SELECT add_compression_policy('features', INTERVAL '7 days');

        -- Retention policy (keep 90 days)
        SELECT add_retention_policy('orderbook_snapshots', INTERVAL '90 days');
        SELECT add_retention_policy('features', INTERVAL '90 days');
        """

    @staticmethod
    def optimized_query_recent_orderbooks(symbol: str, limit: int = 100):
        """Optimized query for recent order books."""
        return f"""
        SELECT timestamp, bids, asks
        FROM orderbook_snapshots
        WHERE symbol = '{symbol}'
        ORDER BY timestamp DESC
        LIMIT {limit};
        """

    @staticmethod
    def optimized_query_features_batch(symbol: str, start_time: str, end_time: str):
        """Optimized batch feature query."""
        return f"""
        SELECT *
        FROM features
        WHERE symbol = '{symbol}'
          AND timestamp BETWEEN '{start_time}' AND '{end_time}'
        ORDER BY timestamp ASC;
        """


class CachingStrategy:
    """Intelligent caching for predictions and features."""

    def __init__(self, redis_client=None):
        """Initialize with optional Redis client."""
        self.redis = redis_client
        self.local_cache = {}

    def cache_prediction(self, symbol: str, timestamp: int, prediction: dict, ttl: int = 60):
        """
        Cache prediction result.

        Args:
            symbol: Trading symbol
            timestamp: Timestamp
            prediction: Prediction dictionary
            ttl: Time to live in seconds
        """
        key = f"pred:{symbol}:{timestamp}"

        if self.redis:
            # Use Redis for distributed caching
            import json
            self.redis.setex(key, ttl, json.dumps(prediction))
        else:
            # Use local cache
            self.local_cache[key] = {
                'data': prediction,
                'expires': time.time() + ttl
            }

    def get_cached_prediction(self, symbol: str, timestamp: int):
        """Get cached prediction if available."""
        key = f"pred:{symbol}:{timestamp}"

        if self.redis:
            import json
            result = self.redis.get(key)
            return json.loads(result) if result else None
        else:
            cached = self.local_cache.get(key)
            if cached and cached['expires'] > time.time():
                return cached['data']
            return None

    def invalidate_cache(self, pattern: str = "*"):
        """Invalidate cache by pattern."""
        if self.redis:
            for key in self.redis.scan_iter(pattern):
                self.redis.delete(key)
        else:
            self.local_cache.clear()


# Performance Benchmarks

def benchmark_feature_calculation():
    """Benchmark feature calculation performance."""
    print("="*80)
    print("FEATURE CALCULATION BENCHMARK")
    print("="*80)

    # Generate synthetic data
    n_snapshots = 1000
    order_books = []

    for _ in range(n_snapshots):
        bids = np.random.rand(10, 2) * 100
        asks = np.random.rand(10, 2) * 100
        asks[:, 0] += 0.1  # Ensure asks > bids

        order_books.append((bids, asks))

    # Benchmark optimized version
    calculator = OptimizedFeatureCalculator()
    features_df = calculator.calculate_features_batch(order_books)

    print(f"\nProcessed {n_snapshots} order books")
    print(f"Features calculated: {len(features_df.columns)}")
    print(f"Throughput: {n_snapshots / (time.time() - time.time()):.0f} snapshots/sec")


def benchmark_model_inference():
    """Benchmark model inference speed."""
    print("\n" + "="*80)
    print("MODEL INFERENCE BENCHMARK")
    print("="*80)

    import torch
    from models.lstm_model import OrderBookLSTM

    # Create model
    model = OrderBookLSTM(
        input_size=20,
        hidden_size=128,
        num_layers=2,
        num_classes=3
    )
    model.eval()

    # Generate batch data
    batch_sizes = [1, 16, 32, 64, 128]

    for batch_size in batch_sizes:
        X = torch.randn(batch_size, 20, 20)

        # Warmup
        with torch.no_grad():
            _ = model(X)

        # Benchmark
        start = time.time()
        iterations = 100

        with torch.no_grad():
            for _ in range(iterations):
                _ = model(X)

        elapsed = time.time() - start
        latency = (elapsed / iterations) * 1000
        throughput = (batch_size * iterations) / elapsed

        print(f"Batch size {batch_size:3d}: {latency:6.2f}ms latency, "
              f"{throughput:7.0f} predictions/sec")


if __name__ == '__main__':
    print("="*80)
    print("PERFORMANCE OPTIMIZATION SUITE")
    print("="*80)

    # Run benchmarks
    benchmark_feature_calculation()
    benchmark_model_inference()

    print("\n" + "="*80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*80)
    print("""
    1. Use Numba JIT for numerical computations (10-100x speedup)
    2. Batch process order books (reduces overhead)
    3. Create TimescaleDB indexes (100x faster queries)
    4. Enable Redis caching (sub-ms latency)
    5. Use connection pooling for database
    6. Implement request batching for API
    7. Enable gzip compression for responses
    8. Use CDN for static assets
    9. Profile regularly with cProfile
    10. Monitor memory with memory_profiler
    """)
    print("="*80)
