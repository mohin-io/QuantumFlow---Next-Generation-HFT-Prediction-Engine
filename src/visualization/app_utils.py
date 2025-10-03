"""
Utility functions for the Streamlit Trading Dashboard.
These functions are designed to be testable and reusable.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px


class OrderBookGenerator:
    """Generate realistic order book data for testing and simulation."""

    @staticmethod
    def generate_orderbook(
        mid_price: float = 50000.0,
        levels: int = 20,
        spread_bps: float = 5.0,
        volatility: float = 100.0,
    ) -> Tuple[List[List[float]], List[List[float]], float]:
        """
        Generate synthetic order book data.

        Args:
            mid_price: Center price for the order book
            levels: Number of price levels on each side
            spread_bps: Spread in basis points
            volatility: Price volatility for randomization

        Returns:
            Tuple of (bids, asks, actual_mid_price)
        """
        actual_mid = mid_price + np.random.normal(0, volatility)
        spread = actual_mid * (spread_bps / 10000)

        bids = []
        asks = []

        for i in range(levels):
            bid_price = actual_mid - spread / 2 - (i * spread / 2)
            ask_price = actual_mid + spread / 2 + (i * spread / 2)
            bid_vol = np.random.uniform(0.5, 5.0) * (1 - i / levels)
            ask_vol = np.random.uniform(0.5, 5.0) * (1 - i / levels)

            bids.append([round(bid_price, 2), round(bid_vol, 4)])
            asks.append([round(ask_price, 2), round(ask_vol, 4)])

        return bids, asks, round(actual_mid, 2)


class MetricsCalculator:
    """Calculate trading and market metrics."""

    @staticmethod
    def calculate_spread_metrics(
        bids: List[List[float]], asks: List[List[float]], mid_price: float
    ) -> Dict[str, float]:
        """Calculate spread-related metrics."""
        if not bids or not asks or mid_price == 0:
            return {"spread": 0.0, "spread_bps": 0.0, "spread_pct": 0.0}

        spread = asks[0][0] - bids[0][0]
        spread_bps = (spread / mid_price) * 10000
        spread_pct = (spread / mid_price) * 100

        return {
            "spread": round(spread, 2),
            "spread_bps": round(spread_bps, 2),
            "spread_pct": round(spread_pct, 4),
        }

    @staticmethod
    def calculate_volume_imbalance(
        bids: List[List[float]], asks: List[List[float]]
    ) -> float:
        """Calculate volume imbalance between bids and asks."""
        if not bids or not asks:
            return 0.0

        total_bid_vol = sum(b[1] for b in bids)
        total_ask_vol = sum(a[1] for a in asks)

        if total_bid_vol + total_ask_vol == 0:
            return 0.0

        imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
        return round(imbalance, 4)

    @staticmethod
    def calculate_sharpe_ratio(
        returns: np.ndarray, risk_free_rate: float = 0.0, periods: int = 252
    ) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - risk_free_rate / periods
        std_returns = np.std(excess_returns)

        if std_returns == 0 or np.isclose(std_returns, 0):
            return 0.0

        sharpe = np.mean(excess_returns) / std_returns * np.sqrt(periods)
        return round(sharpe, 3)

    @staticmethod
    def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown from cumulative returns."""
        if len(cumulative_returns) < 2:
            return 0.0

        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_dd = np.min(drawdown)
        return round(max_dd, 4)

    @staticmethod
    def calculate_win_rate(predictions: List[str], actuals: List[str]) -> float:
        """Calculate prediction win rate."""
        if not predictions or not actuals or len(predictions) != len(actuals):
            return 0.0

        correct = sum(p == a for p, a in zip(predictions, actuals))
        win_rate = correct / len(predictions)
        return round(win_rate, 4)


class SignalGenerator:
    """Generate trading signals based on order book features."""

    @staticmethod
    def generate_signal(
        volume_imbalance: float, spread_bps: float, threshold: float = 0.1
    ) -> Tuple[str, float]:
        """
        Generate trading signal based on volume imbalance.

        Args:
            volume_imbalance: Volume imbalance metric
            spread_bps: Spread in basis points
            threshold: Imbalance threshold for signal generation

        Returns:
            Tuple of (signal, confidence)
        """
        # Simple rule-based signal
        if abs(volume_imbalance) < threshold:
            return "FLAT", 0.5 + abs(volume_imbalance) / threshold * 0.2

        if volume_imbalance > threshold:
            confidence = min(0.95, 0.6 + abs(volume_imbalance) * 2)
            return "UP", round(confidence, 3)
        else:
            confidence = min(0.95, 0.6 + abs(volume_imbalance) * 2)
            return "DOWN", round(confidence, 3)

    @staticmethod
    def calculate_signal_probabilities(
        signal: str, confidence: float
    ) -> Dict[str, float]:
        """Convert signal to probability distribution."""
        probabilities = {"UP": 0.33, "FLAT": 0.34, "DOWN": 0.33}

        if signal == "UP":
            probabilities["UP"] = confidence
            probabilities["FLAT"] = (1 - confidence) * 0.5
            probabilities["DOWN"] = (1 - confidence) * 0.5
        elif signal == "DOWN":
            probabilities["DOWN"] = confidence
            probabilities["FLAT"] = (1 - confidence) * 0.5
            probabilities["UP"] = (1 - confidence) * 0.5
        else:  # FLAT
            probabilities["FLAT"] = confidence
            probabilities["UP"] = (1 - confidence) * 0.5
            probabilities["DOWN"] = (1 - confidence) * 0.5

        # Normalize
        total = sum(probabilities.values())
        probabilities = {k: round(v / total, 4) for k, v in probabilities.items()}

        return probabilities


class DataGenerator:
    """Generate time-series data for visualization."""

    @staticmethod
    def generate_price_history(
        current_price: float, periods: int = 100, volatility: float = 0.001
    ) -> pd.DataFrame:
        """Generate realistic price history using geometric Brownian motion."""
        returns = np.random.normal(0, volatility, periods)
        prices = current_price * np.exp(np.cumsum(returns))

        timestamps = pd.date_range(end=datetime.now(), periods=periods, freq="1s")

        return pd.DataFrame({"timestamp": timestamps, "price": prices})

    @staticmethod
    def generate_prediction_history(
        periods: int = 50,
    ) -> pd.DataFrame:
        """Generate sample prediction history."""
        predictions = np.random.choice(["UP", "DOWN", "FLAT"], periods)
        confidences = np.random.uniform(0.5, 0.95, periods)
        actuals = np.random.choice(["UP", "DOWN", "FLAT"], periods)

        # Make predictions somewhat correlated with actuals (realistic)
        for i in range(periods):
            if np.random.random() < 0.6:  # 60% accuracy
                actuals[i] = predictions[i]

        timestamps = pd.date_range(end=datetime.now(), periods=periods, freq="5s")

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "prediction": predictions,
                "confidence": np.round(confidences, 3),
                "actual": actuals,
            }
        )

    @staticmethod
    def generate_pnl_curve(
        periods: int = 1000, sharpe_target: float = 1.5
    ) -> pd.DataFrame:
        """Generate PnL curve targeting specific Sharpe ratio."""
        # Generate returns with target Sharpe
        mean_return = 0.0005
        std_return = mean_return / (sharpe_target / np.sqrt(252))

        returns = np.random.normal(mean_return, std_return, periods)
        cumulative = np.cumprod(1 + returns)

        timestamps = pd.date_range(end=datetime.now(), periods=periods, freq="1min")

        return pd.DataFrame({"timestamp": timestamps, "pnl": cumulative})


class PlotGenerator:
    """Generate Plotly visualizations."""

    @staticmethod
    def create_orderbook_plot(
        bids: List[List[float]], asks: List[List[float]]
    ) -> go.Figure:
        """Create order book depth visualization."""
        fig = go.Figure()

        # Bid side (green)
        bid_prices = [b[0] for b in bids]
        bid_vols = [b[1] for b in bids]

        fig.add_trace(
            go.Bar(
                x=bid_vols,
                y=bid_prices,
                orientation="h",
                name="Bids",
                marker=dict(color="green", opacity=0.7),
            )
        )

        # Ask side (red)
        ask_prices = [a[0] for a in asks]
        ask_vols = [-a[1] for a in asks]

        fig.add_trace(
            go.Bar(
                x=ask_vols,
                y=ask_prices,
                orientation="h",
                name="Asks",
                marker=dict(color="red", opacity=0.7),
            )
        )

        fig.update_layout(
            title="Order Book Depth",
            xaxis_title="Volume",
            yaxis_title="Price",
            height=500,
            barmode="relative",
            showlegend=True,
        )

        return fig

    @staticmethod
    def create_signal_probability_plot(probabilities: Dict[str, float]) -> go.Figure:
        """Create signal probability bar chart."""
        colors = {
            "UP": "green",
            "FLAT": "gray",
            "DOWN": "red",
        }

        fig = go.Figure(
            data=[
                go.Bar(
                    x=list(probabilities.keys()),
                    y=list(probabilities.values()),
                    marker_color=[colors[k] for k in probabilities.keys()],
                )
            ]
        )

        fig.update_layout(
            title="Signal Probabilities",
            yaxis_title="Probability",
            height=300,
            yaxis_range=[0, 1],
        )

        return fig

    @staticmethod
    def create_performance_metrics_plot(metrics: Dict[str, float]) -> go.Figure:
        """Create performance metrics bar chart."""
        fig = px.bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            title="Classification Performance",
            labels={"x": "Metric", "y": "Value"},
        )

        fig.update_layout(yaxis_range=[0, 1], height=400)

        return fig


class FeatureAnalyzer:
    """Analyze and visualize features."""

    @staticmethod
    def generate_feature_importance(
        n_features: int = 9,
    ) -> pd.DataFrame:
        """Generate sample feature importance data."""
        features = [
            "OFI_L1",
            "OFI_L5",
            "OFI_L10",
            "Micro_Price_Dev",
            "Volume_Imbalance",
            "Spread_BPS",
            "Realized_Vol_20",
            "Depth_Imbalance",
            "Liquidity_Conc",
        ][:n_features]

        # Generate importance with some structure (first features more important)
        importance = np.random.beta(2, 5, n_features)
        importance = importance / importance.sum()

        return pd.DataFrame(
            {"Feature": features, "Importance": importance}
        ).sort_values("Importance", ascending=False)

    @staticmethod
    def generate_correlation_matrix(
        features: List[str],
    ) -> np.ndarray:
        """Generate realistic correlation matrix."""
        n = len(features)
        corr_matrix = np.random.uniform(-0.3, 0.3, (n, n))
        np.fill_diagonal(corr_matrix, 1.0)

        # Make symmetric
        corr_matrix = (corr_matrix + corr_matrix.T) / 2

        return corr_matrix


# Validation functions
def validate_orderbook_data(
    bids: List[List[float]], asks: List[List[float]]
) -> bool:
    """Validate order book data structure."""
    if not bids or not asks:
        return False

    # Check that bids are decreasing
    bid_prices = [b[0] for b in bids]
    if bid_prices != sorted(bid_prices, reverse=True):
        return False

    # Check that asks are increasing
    ask_prices = [a[0] for a in asks]
    if ask_prices != sorted(ask_prices):
        return False

    # Check that all volumes are positive
    if any(b[1] <= 0 for b in bids) or any(a[1] <= 0 for a in asks):
        return False

    # Check that there's no overlap (best bid < best ask)
    if bids[0][0] >= asks[0][0]:
        return False

    return True


def format_currency(value: float, decimals: int = 2) -> str:
    """Format number as currency."""
    return f"${value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format number as percentage."""
    return f"{value * 100:.{decimals}f}%"
