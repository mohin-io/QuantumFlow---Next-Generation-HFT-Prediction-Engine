"""
Order Flow Imbalance (OFI) Calculator

OFI measures the imbalance between buy and sell pressure in the limit order book.
It's a powerful predictor of short-term price movements.

Mathematical Definition:
OFI(t) = Σ[i=1 to N] [I(ΔV_bid^i > 0) × ΔV_bid^i - I(ΔV_ask^i > 0) × ΔV_ask^i]

Where:
- ΔV_bid^i: Change in bid volume at level i
- ΔV_ask^i: Change in ask volume at level i
- N: Number of price levels considered
- I(): Indicator function (1 if condition true, 0 otherwise)

Reference:
Cont, R., Kukanov, A., & Stoikov, S. (2014). The price impact of order book events.
Journal of Financial Econometrics, 12(1), 47-88.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class OrderBookState:
    """Represents the state of an order book at a point in time."""

    timestamp: float
    bids: List[Tuple[float, float]]  # [(price, volume), ...]
    asks: List[Tuple[float, float]]  # [(price, volume), ...]

    def get_bid_volume_at_level(self, level: int) -> float:
        """Get bid volume at specified level (0-indexed)."""
        return self.bids[level][1] if level < len(self.bids) else 0.0

    def get_ask_volume_at_level(self, level: int) -> float:
        """Get ask volume at specified level (0-indexed)."""
        return self.asks[level][1] if level < len(self.asks) else 0.0

    def get_bid_price_at_level(self, level: int) -> float:
        """Get bid price at specified level (0-indexed)."""
        return self.bids[level][0] if level < len(self.bids) else 0.0

    def get_ask_price_at_level(self, level: int) -> float:
        """Get ask price at specified level (0-indexed)."""
        return self.asks[level][0] if level < len(self.asks) else 0.0


class OrderFlowImbalanceCalculator:
    """
    Calculator for Order Flow Imbalance (OFI) and related metrics.

    Maintains a sliding window of order book states to compute:
    - OFI at different depth levels
    - Cumulative OFI
    - OFI momentum
    """

    def __init__(
        self,
        levels: List[int] = [1, 5, 10],
        window_sizes: List[int] = [10, 50, 100],
        use_price_levels: bool = False
    ):
        """
        Initialize OFI calculator.

        Args:
            levels: Price levels to compute OFI for (1-indexed, e.g., [1, 5, 10])
            window_sizes: Window sizes for rolling OFI metrics
            use_price_levels: If True, track by price level instead of position
        """
        self.levels = levels
        self.window_sizes = window_sizes
        self.use_price_levels = use_price_levels

        # Store previous order book state
        self.prev_state: Optional[OrderBookState] = None

        # Rolling windows for OFI values
        self.ofi_windows = {
            level: {
                window_size: deque(maxlen=window_size)
                for window_size in window_sizes
            }
            for level in levels
        }

    def compute_ofi(
        self,
        current_state: OrderBookState,
        num_levels: int = 10
    ) -> Dict[str, float]:
        """
        Compute Order Flow Imbalance for the current state.

        Args:
            current_state: Current order book state
            num_levels: Number of levels to consider

        Returns:
            Dictionary of OFI metrics
        """
        if self.prev_state is None:
            # First observation, no OFI to compute
            self.prev_state = current_state
            return self._initialize_ofi_dict()

        ofi_values = {}

        # Compute OFI for each specified level
        for max_level in self.levels:
            ofi = 0.0

            for level in range(min(max_level, num_levels)):
                # Get current and previous volumes
                curr_bid_vol = current_state.get_bid_volume_at_level(level)
                curr_ask_vol = current_state.get_ask_volume_at_level(level)

                prev_bid_vol = self.prev_state.get_bid_volume_at_level(level)
                prev_ask_vol = self.prev_state.get_ask_volume_at_level(level)

                # Compute volume changes
                delta_bid = curr_bid_vol - prev_bid_vol
                delta_ask = curr_ask_vol - prev_ask_vol

                # OFI contribution from this level
                # Positive bid volume change adds to OFI
                # Positive ask volume change subtracts from OFI
                if delta_bid > 0:
                    ofi += delta_bid
                if delta_ask > 0:
                    ofi -= delta_ask

            # Store OFI for this level
            ofi_values[f'ofi_L{max_level}'] = ofi

            # Update rolling windows
            for window_size in self.window_sizes:
                self.ofi_windows[max_level][window_size].append(ofi)

                # Compute rolling statistics
                window_data = list(self.ofi_windows[max_level][window_size])
                if len(window_data) > 0:
                    ofi_values[f'ofi_L{max_level}_mean_{window_size}'] = np.mean(window_data)
                    ofi_values[f'ofi_L{max_level}_std_{window_size}'] = np.std(window_data)
                    ofi_values[f'ofi_L{max_level}_sum_{window_size}'] = np.sum(window_data)

        # Update previous state
        self.prev_state = current_state

        return ofi_values

    def compute_signed_ofi(
        self,
        current_state: OrderBookState,
        num_levels: int = 10
    ) -> Dict[str, float]:
        """
        Compute signed OFI that accounts for price level changes.

        This version is more sophisticated and considers whether volume
        additions/deletions occur at the same price level.
        """
        if self.prev_state is None:
            self.prev_state = current_state
            return self._initialize_ofi_dict()

        ofi_values = {}

        for max_level in self.levels:
            bid_ofi = 0.0
            ask_ofi = 0.0

            for level in range(min(max_level, num_levels)):
                curr_bid_price = current_state.get_bid_price_at_level(level)
                curr_bid_vol = current_state.get_bid_volume_at_level(level)
                prev_bid_price = self.prev_state.get_bid_price_at_level(level)
                prev_bid_vol = self.prev_state.get_bid_volume_at_level(level)

                curr_ask_price = current_state.get_ask_price_at_level(level)
                curr_ask_vol = current_state.get_ask_volume_at_level(level)
                prev_ask_price = self.prev_state.get_ask_price_at_level(level)
                prev_ask_vol = self.prev_state.get_ask_volume_at_level(level)

                # Bid side OFI
                if curr_bid_price == prev_bid_price:
                    # Same price level, measure volume change
                    delta_bid = curr_bid_vol - prev_bid_vol
                    if delta_bid > 0:
                        bid_ofi += delta_bid
                elif curr_bid_price > prev_bid_price:
                    # Price improved, add full volume
                    bid_ofi += curr_bid_vol

                # Ask side OFI
                if curr_ask_price == prev_ask_price:
                    # Same price level, measure volume change
                    delta_ask = curr_ask_vol - prev_ask_vol
                    if delta_ask > 0:
                        ask_ofi += delta_ask
                elif curr_ask_price < prev_ask_price:
                    # Price improved, add full volume
                    ask_ofi += curr_ask_vol

            # Net OFI
            net_ofi = bid_ofi - ask_ofi
            ofi_values[f'signed_ofi_L{max_level}'] = net_ofi
            ofi_values[f'bid_ofi_L{max_level}'] = bid_ofi
            ofi_values[f'ask_ofi_L{max_level}'] = ask_ofi

        self.prev_state = current_state
        return ofi_values

    def _initialize_ofi_dict(self) -> Dict[str, float]:
        """Initialize OFI dictionary with zeros."""
        ofi_dict = {}
        for level in self.levels:
            ofi_dict[f'ofi_L{level}'] = 0.0
            for window_size in self.window_sizes:
                ofi_dict[f'ofi_L{level}_mean_{window_size}'] = 0.0
                ofi_dict[f'ofi_L{level}_std_{window_size}'] = 0.0
                ofi_dict[f'ofi_L{level}_sum_{window_size}'] = 0.0
        return ofi_dict

    def reset(self):
        """Reset the calculator state."""
        self.prev_state = None
        for level in self.levels:
            for window_size in self.window_sizes:
                self.ofi_windows[level][window_size].clear()


def compute_ofi_from_dataframe(
    df: pd.DataFrame,
    levels: List[int] = [1, 5, 10],
    window_sizes: List[int] = [10, 50, 100]
) -> pd.DataFrame:
    """
    Compute OFI features from a DataFrame of order book snapshots.

    Args:
        df: DataFrame with columns: timestamp, bids, asks
        levels: Price levels to compute OFI for
        window_sizes: Window sizes for rolling metrics

    Returns:
        DataFrame with OFI features added
    """
    calculator = OrderFlowImbalanceCalculator(levels=levels, window_sizes=window_sizes)

    ofi_features = []

    for idx, row in df.iterrows():
        # Convert bids and asks to list of tuples
        bids = [(b[0], b[1]) for b in row['bids']] if isinstance(row['bids'], list) else []
        asks = [(a[0], a[1]) for a in row['asks']] if isinstance(row['asks'], list) else []

        state = OrderBookState(
            timestamp=row['timestamp'],
            bids=bids,
            asks=asks
        )

        ofi_vals = calculator.compute_ofi(state)
        ofi_features.append(ofi_vals)

    # Convert to DataFrame and merge with original
    ofi_df = pd.DataFrame(ofi_features)
    result_df = pd.concat([df.reset_index(drop=True), ofi_df], axis=1)

    return result_df


# Example usage and testing
if __name__ == "__main__":
    # Example: Generate synthetic order book data
    np.random.seed(42)

    # Simulate order book snapshots
    num_snapshots = 1000
    snapshots = []

    mid_price = 100.0
    for i in range(num_snapshots):
        # Random walk for mid price
        mid_price += np.random.normal(0, 0.01)

        # Generate bids and asks
        bids = []
        asks = []
        for level in range(10):
            bid_price = mid_price - 0.01 * (level + 1)
            ask_price = mid_price + 0.01 * (level + 1)
            bid_volume = np.random.uniform(10, 100)
            ask_volume = np.random.uniform(10, 100)

            bids.append([bid_price, bid_volume])
            asks.append([ask_price, ask_volume])

        snapshots.append({
            'timestamp': i,
            'bids': bids,
            'asks': asks
        })

    # Create DataFrame
    df = pd.DataFrame(snapshots)

    # Compute OFI features
    print("Computing Order Flow Imbalance features...")
    df_with_ofi = compute_ofi_from_dataframe(df, levels=[1, 5, 10], window_sizes=[10, 50])

    # Display results
    print("\n" + "="*80)
    print("Order Flow Imbalance Features (first 10 rows)")
    print("="*80)

    ofi_columns = [col for col in df_with_ofi.columns if 'ofi' in col]
    print(df_with_ofi[['timestamp'] + ofi_columns[:8]].head(10))

    print("\n" + "="*80)
    print("OFI Statistics")
    print("="*80)
    print(df_with_ofi[ofi_columns].describe())

    # Visualize OFI if matplotlib available
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        axes[0].plot(df_with_ofi['timestamp'], df_with_ofi['ofi_L1'], label='OFI L1', alpha=0.7)
        axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.3)
        axes[0].set_title('Order Flow Imbalance - Level 1')
        axes[0].set_xlabel('Tick')
        axes[0].set_ylabel('OFI')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(df_with_ofi['timestamp'], df_with_ofi['ofi_L5'], label='OFI L5', alpha=0.7, color='orange')
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.3)
        axes[1].set_title('Order Flow Imbalance - Level 5')
        axes[1].set_xlabel('Tick')
        axes[1].set_ylabel('OFI')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        axes[2].plot(df_with_ofi['timestamp'], df_with_ofi['ofi_L10'], label='OFI L10', alpha=0.7, color='green')
        axes[2].axhline(y=0, color='r', linestyle='--', alpha=0.3)
        axes[2].set_title('Order Flow Imbalance - Level 10')
        axes[2].set_xlabel('Tick')
        axes[2].set_ylabel('OFI')
        axes[2].legend()
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('data/simulations/ofi_example.png', dpi=150)
        print("\nVisualization saved to: data/simulations/ofi_example.png")

    except ImportError:
        print("\nMatplotlib not available for visualization")
