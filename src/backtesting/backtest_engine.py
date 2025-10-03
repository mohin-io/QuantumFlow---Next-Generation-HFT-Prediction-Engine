"""
Backtesting Engine for Order Book Imbalance Trading Strategy

Simulates trading based on model predictions and calculates PnL metrics.

Key Metrics:
- Total Return / Sharpe Ratio
- Maximum Drawdown
- Win Rate / Profit Factor
- Transaction Cost Impact
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: pd.Timestamp
    direction: int  # 1=long, -1=short, 0=neutral
    entry_price: float
    exit_price: Optional[float] = None
    exit_timestamp: Optional[pd.Timestamp] = None
    quantity: float = 1.0
    pnl: Optional[float] = None
    fees: float = 0.0


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_capital: float = 100000.0
    position_size: float = 0.1  # Fraction of capital per trade
    transaction_cost_bps: float = 5.0  # Basis points (0.05%)
    slippage_bps: float = 2.0  # Basis points
    max_holding_period: int = 100  # Maximum ticks to hold position
    confidence_threshold: float = 0.6  # Min probability to trade


class BacktestEngine:
    """
    Backtesting engine for order book imbalance strategies.

    Workflow:
    1. Load predictions and price data
    2. Generate trading signals
    3. Simulate order execution with realistic costs
    4. Calculate performance metrics
    """

    def __init__(self, config: BacktestConfig = None):
        """Initialize backtest engine."""
        self.config = config or BacktestConfig()
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.positions: Dict[str, Trade] = {}  # Active positions
        self.capital = self.config.initial_capital

    def run(self,
            predictions_df: pd.DataFrame,
            prices_df: pd.DataFrame) -> Dict:
        """
        Run backtest on predictions.

        Args:
            predictions_df: DataFrame with columns [timestamp, prediction, probability]
            prices_df: DataFrame with columns [timestamp, mid_price, bid, ask]

        Returns:
            Dictionary with performance metrics
        """
        # Merge predictions with prices
        df = pd.merge_asof(
            predictions_df.sort_values('timestamp'),
            prices_df.sort_values('timestamp'),
            on='timestamp',
            direction='nearest'
        )

        # Initialize
        self.trades = []
        self.equity_curve = [self.config.initial_capital]
        self.positions = {}
        self.capital = self.config.initial_capital
        current_position = 0

        # Simulate trading
        for idx, row in df.iterrows():
            timestamp = row['timestamp']
            prediction = row['prediction']  # 0=down, 1=neutral, 2=up
            probability = row.get('probability', 0.5)
            mid_price = row['mid_price']
            bid_price = row.get('bid', mid_price * 0.9995)
            ask_price = row.get('ask', mid_price * 1.0005)

            # Generate signal
            signal = self._generate_signal(prediction, probability)

            # Close existing position if needed
            if current_position != 0 and signal != current_position:
                self._close_position(timestamp, mid_price, bid_price, ask_price)
                current_position = 0

            # Open new position
            if signal != 0 and current_position == 0:
                self._open_position(timestamp, signal, mid_price, bid_price, ask_price)
                current_position = signal

            # Update equity curve
            unrealized_pnl = self._calculate_unrealized_pnl(mid_price)
            equity = self.capital + unrealized_pnl
            self.equity_curve.append(equity)

        # Close any remaining positions
        if self.positions:
            final_price = df.iloc[-1]['mid_price']
            final_bid = df.iloc[-1].get('bid', final_price * 0.9995)
            final_ask = df.iloc[-1].get('ask', final_price * 1.0005)
            self._close_position(df.iloc[-1]['timestamp'], final_price, final_bid, final_ask)

        # Calculate metrics
        metrics = self._calculate_metrics()
        return metrics

    def _generate_signal(self, prediction: int, probability: float) -> int:
        """
        Generate trading signal from prediction.

        Returns:
            1 (long), -1 (short), or 0 (neutral)
        """
        if probability < self.config.confidence_threshold:
            return 0

        if prediction == 2:  # Up prediction
            return 1
        elif prediction == 0:  # Down prediction
            return -1
        else:
            return 0

    def _open_position(self, timestamp, direction, mid_price, bid_price, ask_price):
        """Open a new position."""
        # Calculate position size
        position_value = self.capital * self.config.position_size

        # Apply slippage
        if direction == 1:  # Long - buy at ask
            execution_price = ask_price * (1 + self.config.slippage_bps / 10000)
        else:  # Short - sell at bid
            execution_price = bid_price * (1 - self.config.slippage_bps / 10000)

        quantity = position_value / execution_price

        # Calculate transaction costs
        fees = position_value * self.config.transaction_cost_bps / 10000

        # Create trade
        trade = Trade(
            timestamp=timestamp,
            direction=direction,
            entry_price=execution_price,
            quantity=quantity,
            fees=fees
        )

        self.positions['current'] = trade
        self.capital -= fees

    def _close_position(self, timestamp, mid_price, bid_price, ask_price):
        """Close active position."""
        if 'current' not in self.positions:
            return

        trade = self.positions.pop('current')

        # Apply slippage
        if trade.direction == 1:  # Close long - sell at bid
            execution_price = bid_price * (1 - self.config.slippage_bps / 10000)
        else:  # Close short - buy at ask
            execution_price = ask_price * (1 + self.config.slippage_bps / 10000)

        # Calculate PnL
        position_value = trade.quantity * trade.entry_price
        exit_value = trade.quantity * execution_price

        if trade.direction == 1:
            pnl = exit_value - position_value
        else:
            pnl = position_value - exit_value

        # Transaction costs
        fees = exit_value * self.config.transaction_cost_bps / 10000
        pnl -= fees
        pnl -= trade.fees  # Entry fees

        # Update trade
        trade.exit_price = execution_price
        trade.exit_timestamp = timestamp
        trade.pnl = pnl

        # Update capital
        self.capital += pnl

        self.trades.append(trade)

    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL for open positions."""
        if 'current' not in self.positions:
            return 0.0

        trade = self.positions['current']
        position_value = trade.quantity * trade.entry_price
        current_value = trade.quantity * current_price

        if trade.direction == 1:
            return current_value - position_value
        else:
            return position_value - current_value

    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if len(self.trades) == 0:
            return {
                'total_trades': 0,
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_trade_pnl': 0.0
            }

        # Extract trade PnLs
        pnls = np.array([t.pnl for t in self.trades])

        # Total return
        total_return = (self.capital - self.config.initial_capital) / self.config.initial_capital

        # Sharpe ratio (annualized, assuming 252 trading days)
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        sharpe = np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-10)

        # Maximum drawdown
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # Win rate
        winning_trades = np.sum(pnls > 0)
        win_rate = winning_trades / len(pnls)

        # Profit factor
        gross_profit = np.sum(pnls[pnls > 0])
        gross_loss = abs(np.sum(pnls[pnls < 0]))
        profit_factor = gross_profit / (gross_loss + 1e-10)

        # Average trade
        avg_trade_pnl = np.mean(pnls)

        return {
            'total_trades': len(self.trades),
            'total_return': total_return * 100,  # Percentage
            'sharpe_ratio': sharpe,
            'max_drawdown': abs(max_drawdown) * 100,  # Percentage
            'win_rate': win_rate * 100,  # Percentage
            'profit_factor': profit_factor,
            'avg_trade_pnl': avg_trade_pnl,
            'final_capital': self.capital,
            'total_pnl': self.capital - self.config.initial_capital
        }

    def get_trades_df(self) -> pd.DataFrame:
        """Return trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                'timestamp': t.timestamp,
                'direction': 'LONG' if t.direction == 1 else 'SHORT',
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'exit_timestamp': t.exit_timestamp,
                'pnl': t.pnl,
                'fees': t.fees
            }
            for t in self.trades
        ])

    def plot_results(self):
        """Plot backtest results."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Equity curve
        axes[0, 0].plot(self.equity_curve)
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_xlabel('Time Steps')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True)

        # PnL distribution
        if self.trades:
            pnls = [t.pnl for t in self.trades]
            axes[0, 1].hist(pnls, bins=30, edgecolor='black')
            axes[0, 1].set_title('PnL Distribution')
            axes[0, 1].set_xlabel('PnL ($)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.5)
            axes[0, 1].grid(True)

        # Drawdown
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max * 100

        axes[1, 0].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3)
        axes[1, 0].set_title('Drawdown')
        axes[1, 0].set_xlabel('Time Steps')
        axes[1, 0].set_ylabel('Drawdown (%)')
        axes[1, 0].grid(True)

        # Cumulative PnL
        if self.trades:
            cumulative_pnl = np.cumsum(pnls)
            axes[1, 1].plot(cumulative_pnl)
            axes[1, 1].set_title('Cumulative PnL')
            axes[1, 1].set_xlabel('Trade Number')
            axes[1, 1].set_ylabel('Cumulative PnL ($)')
            axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
            axes[1, 1].grid(True)

        plt.tight_layout()
        return fig


# Demonstration
if __name__ == "__main__":
    print("="*80)
    print("BACKTESTING ENGINE DEMONSTRATION")
    print("="*80)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000

    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='1min')

    # Synthetic price data with trend
    price = 100.0
    prices = []
    for _ in range(n_samples):
        price += np.random.randn() * 0.1
        prices.append(price)

    prices_df = pd.DataFrame({
        'timestamp': timestamps,
        'mid_price': prices,
        'bid': np.array(prices) * 0.9995,
        'ask': np.array(prices) * 1.0005
    })

    # Synthetic predictions (with some signal)
    true_direction = np.diff(prices) > 0
    predictions = []
    probabilities = []

    for i in range(len(true_direction)):
        # 65% accuracy
        if np.random.rand() < 0.65:
            pred = 2 if true_direction[i] else 0
            prob = 0.7
        else:
            pred = 0 if true_direction[i] else 2
            prob = 0.6

        predictions.append(pred)
        probabilities.append(prob)

    predictions_df = pd.DataFrame({
        'timestamp': timestamps[1:],
        'prediction': predictions,
        'probability': probabilities
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

    # Display results
    print("\nBacktest Results:")
    print("-"*80)
    print(f"Total Trades:      {metrics['total_trades']}")
    print(f"Total Return:      {metrics['total_return']:.2f}%")
    print(f"Sharpe Ratio:      {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:      {metrics['max_drawdown']:.2f}%")
    print(f"Win Rate:          {metrics['win_rate']:.2f}%")
    print(f"Profit Factor:     {metrics['profit_factor']:.2f}")
    print(f"Avg Trade PnL:     ${metrics['avg_trade_pnl']:.2f}")
    print(f"Final Capital:     ${metrics['final_capital']:.2f}")
    print(f"Total PnL:         ${metrics['total_pnl']:.2f}")

    print("\n" + "="*80)
    print("Backtest engine ready for production use!")
    print("="*80)
