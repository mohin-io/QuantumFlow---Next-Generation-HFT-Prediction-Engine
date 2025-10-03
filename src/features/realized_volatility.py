"""
Realized Volatility Estimators for High-Frequency Data

Implements various volatility estimators that are suitable for
high-frequency order book data:

1. Simple Realized Volatility (RV)
2. Parkinson Volatility (uses high-low range)
3. Garman-Klass Volatility (open-high-low-close)
4. Yang-Zhang Volatility (accounts for overnight jumps)
5. Rogers-Satchell Volatility (drift-independent)

These estimators provide better volatility estimates than rolling standard
deviation, especially important for short-term trading strategies.

References:
- Parkinson, M. (1980). The Extreme Value Method for Estimating the Variance of the Rate of Return
- Garman, M. B., & Klass, M. J. (1980). On the Estimation of Security Price Volatilities
- Yang, D., & Zhang, Q. (2000). Drift-independent volatility estimation
- Rogers, L. C. G., & Satchell, S. E. (1991). Estimating variance from high, low and closing prices
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional
from dataclasses import dataclass


@dataclass
class VolatilityMetrics:
    """Container for volatility metrics."""

    realized_volatility: float
    parkinson_volatility: float
    garman_klass_volatility: float
    rogers_satchell_volatility: float

    # Annualized versions (assuming 252 trading days, 24h trading for crypto)
    annualized_rv: float
    annualized_parkinson: float


class RealizedVolatilityCalculator:
    """
    Calculator for various realized volatility estimators.

    All estimators return volatility in the same units as the input data.
    """

    @staticmethod
    def simple_realized_volatility(returns: np.ndarray, window: int = 20) -> np.ndarray:
        """
        Simple realized volatility (standard deviation of returns).

        Args:
            returns: Array of returns
            window: Rolling window size

        Returns:
            Array of realized volatility estimates
        """
        return pd.Series(returns).rolling(window=window).std().values

    @staticmethod
    def parkinson_volatility(
        high: np.ndarray, low: np.ndarray, window: int = 20
    ) -> np.ndarray:
        """
        Parkinson volatility estimator.

        Uses high-low range, which is more efficient than close-to-close.

        Formula:
        σ_P = sqrt((1/(4*ln(2))) * (1/n) * Σ[ln(H_i/L_i)]^2)

        Args:
            high: Array of high prices
            low: Array of low prices
            window: Rolling window size

        Returns:
            Array of Parkinson volatility estimates
        """
        hl_ratio = np.log(high / low)
        hl_ratio_sq = hl_ratio**2

        # Rolling mean of squared log ratios
        rolling_mean = pd.Series(hl_ratio_sq).rolling(window=window).mean()

        # Parkinson estimator
        parkinson_vol = np.sqrt(rolling_mean / (4 * np.log(2)))

        return parkinson_vol.values

    @staticmethod
    def garman_klass_volatility(
        open_price: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        window: int = 20,
    ) -> np.ndarray:
        """
        Garman-Klass volatility estimator.

        Uses open, high, low, close prices for better estimation.

        Formula:
        σ_GK = sqrt((1/n) * Σ[0.5*(ln(H/L))^2 - (2*ln(2)-1)*(ln(C/O))^2])

        Args:
            open_price: Array of open prices
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            window: Rolling window size

        Returns:
            Array of Garman-Klass volatility estimates
        """
        hl = np.log(high / low)
        co = np.log(close / open_price)

        component = 0.5 * (hl**2) - (2 * np.log(2) - 1) * (co**2)

        rolling_mean = pd.Series(component).rolling(window=window).mean()

        gk_vol = np.sqrt(rolling_mean)

        return gk_vol.values

    @staticmethod
    def rogers_satchell_volatility(
        open_price: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        window: int = 20,
    ) -> np.ndarray:
        """
        Rogers-Satchell volatility estimator.

        Drift-independent estimator.

        Formula:
        σ_RS = sqrt((1/n) * Σ[ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)])

        Args:
            open_price: Array of open prices
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            window: Rolling window size

        Returns:
            Array of Rogers-Satchell volatility estimates
        """
        hc = np.log(high / close)
        ho = np.log(high / open_price)
        lc = np.log(low / close)
        lo = np.log(low / open_price)

        component = hc * ho + lc * lo

        rolling_mean = pd.Series(component).rolling(window=window).mean()

        rs_vol = np.sqrt(rolling_mean)

        return rs_vol.values

    @staticmethod
    def yang_zhang_volatility(
        open_price: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        window: int = 20,
    ) -> np.ndarray:
        """
        Yang-Zhang volatility estimator.

        Combines overnight and intraday volatility.

        This is one of the most accurate volatility estimators.

        Args:
            open_price: Array of open prices
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            window: Rolling window size

        Returns:
            Array of Yang-Zhang volatility estimates
        """
        # Overnight volatility (close to open)
        overnight_ret = np.log(open_price[1:] / close[:-1])
        overnight_vol = pd.Series(overnight_ret).rolling(window=window).std()

        # Open to close volatility
        oc_ret = np.log(close / open_price)
        oc_vol = pd.Series(oc_ret).rolling(window=window).std()

        # Rogers-Satchell component
        rs_vol = RealizedVolatilityCalculator.rogers_satchell_volatility(
            open_price, high, low, close, window
        )

        # Combine components (simplified version)
        k = 0.34 / (1.34 + (window + 1) / (window - 1))

        # Pad overnight_vol to match length
        overnight_vol = np.concatenate([[np.nan], overnight_vol.values])

        yz_vol = np.sqrt(overnight_vol**2 + k * (oc_vol**2) + (1 - k) * (rs_vol**2))

        return yz_vol


def compute_ohlc_from_snapshots(
    snapshots: List[Dict], window_ticks: int = 20
) -> pd.DataFrame:
    """
    Compute OHLC bars from order book snapshots.

    Args:
        snapshots: List of order book snapshots
        window_ticks: Number of ticks per bar

    Returns:
        DataFrame with OHLC data
    """
    mid_prices = []
    timestamps = []

    for snapshot in snapshots:
        bids = snapshot.get("bids", [])
        asks = snapshot.get("asks", [])

        if bids and asks:
            mid_price = (bids[0][0] + asks[0][0]) / 2
            mid_prices.append(mid_price)
            timestamps.append(snapshot.get("timestamp", 0))

    # Create bars
    bars = []
    for i in range(0, len(mid_prices), window_ticks):
        window_prices = mid_prices[i : i + window_ticks]

        if len(window_prices) > 0:
            bar = {
                "timestamp": timestamps[i],
                "open": window_prices[0],
                "high": max(window_prices),
                "low": min(window_prices),
                "close": window_prices[-1],
            }
            bars.append(bar)

    return pd.DataFrame(bars)


def compute_volatility_features(
    df: pd.DataFrame, windows: List[int] = [20, 50, 100], price_col: str = "close"
) -> pd.DataFrame:
    """
    Compute multiple volatility features from OHLC data.

    Args:
        df: DataFrame with OHLC columns
        windows: List of window sizes
        price_col: Column to use for returns calculation

    Returns:
        DataFrame with volatility features added
    """
    result_df = df.copy()

    # Compute returns
    returns = np.log(df[price_col] / df[price_col].shift(1))

    for window in windows:
        # Simple realized volatility
        rv = RealizedVolatilityCalculator.simple_realized_volatility(
            returns.values, window
        )
        result_df[f"rv_{window}"] = rv

        # Parkinson volatility
        if "high" in df.columns and "low" in df.columns:
            park_vol = RealizedVolatilityCalculator.parkinson_volatility(
                df["high"].values, df["low"].values, window
            )
            result_df[f"parkinson_vol_{window}"] = park_vol

        # Garman-Klass volatility
        if all(col in df.columns for col in ["open", "high", "low", "close"]):
            gk_vol = RealizedVolatilityCalculator.garman_klass_volatility(
                df["open"].values,
                df["high"].values,
                df["low"].values,
                df["close"].values,
                window,
            )
            result_df[f"garman_klass_vol_{window}"] = gk_vol

            # Rogers-Satchell volatility
            rs_vol = RealizedVolatilityCalculator.rogers_satchell_volatility(
                df["open"].values,
                df["high"].values,
                df["low"].values,
                df["close"].values,
                window,
            )
            result_df[f"rogers_satchell_vol_{window}"] = rs_vol

    return result_df


# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic price data
    np.random.seed(42)

    n_periods = 1000
    base_price = 50000
    volatility = 0.02

    # Simulate price path with stochastic volatility
    returns = np.random.normal(0, volatility, n_periods)
    prices = base_price * np.exp(np.cumsum(returns))

    # Create OHLC bars
    bars = []
    bar_size = 20

    for i in range(0, len(prices), bar_size):
        window = prices[i : i + bar_size]
        if len(window) > 0:
            bars.append(
                {
                    "timestamp": i,
                    "open": window[0],
                    "high": np.max(window),
                    "low": np.min(window),
                    "close": window[-1],
                }
            )

    df = pd.DataFrame(bars)

    # Compute volatility features
    print("Computing realized volatility features...")
    df_with_vol = compute_volatility_features(df, windows=[20, 50], price_col="close")

    print("\n" + "=" * 80)
    print("Realized Volatility Features (first 10 rows)")
    print("=" * 80)

    vol_cols = [col for col in df_with_vol.columns if "vol" in col or "rv" in col]
    print(df_with_vol[["timestamp", "close"] + vol_cols[:4]].head(10))

    print("\n" + "=" * 80)
    print("Volatility Statistics")
    print("=" * 80)
    print(df_with_vol[vol_cols].describe())

    # Compare estimators
    print("\n" + "=" * 80)
    print("Volatility Estimator Comparison (average values)")
    print("=" * 80)

    comparison = {}
    for col in vol_cols:
        comparison[col] = df_with_vol[col].mean()

    for name, value in comparison.items():
        print(f"  {name:30s}: {value:.6f}")

    # Visualize
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Price
        axes[0].plot(
            df_with_vol["timestamp"],
            df_with_vol["close"],
            label="Close Price",
            alpha=0.7,
        )
        axes[0].set_title("Price Evolution")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Price")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Volatility comparison
        axes[1].plot(
            df_with_vol["timestamp"],
            df_with_vol["rv_20"],
            label="Simple RV (20)",
            alpha=0.7,
        )
        axes[1].plot(
            df_with_vol["timestamp"],
            df_with_vol["parkinson_vol_20"],
            label="Parkinson (20)",
            alpha=0.7,
        )
        axes[1].plot(
            df_with_vol["timestamp"],
            df_with_vol["garman_klass_vol_20"],
            label="Garman-Klass (20)",
            alpha=0.7,
        )

        axes[1].set_title("Volatility Estimators Comparison")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Volatility")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("data/simulations/realized_volatility_example.png", dpi=150)
        print(
            "\nVisualization saved to: data/simulations/realized_volatility_example.png"
        )

    except ImportError:
        print("\nMatplotlib not available for visualization")
