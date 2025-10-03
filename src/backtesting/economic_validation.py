"""
Economic Validation Framework for Trading Strategies

Validates that trading strategies make economic sense:
- Transaction cost impact analysis
- Market impact modeling
- Realistic execution simulation
- Risk-adjusted performance metrics
- Statistical significance testing

References:
- Campbell, J. Y., Lo, A. W., & MacKinlay, A. C. (1997). The Econometrics of Financial Markets
- Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import scipy.stats as stats
from collections import defaultdict


@dataclass
class MarketConditions:
    """Realistic market microstructure parameters."""

    # Transaction costs
    maker_fee_bps: float = 1.0  # Maker fee (rebate if negative)
    taker_fee_bps: float = 5.0  # Taker fee

    # Market impact (square-root model: MI = λ * sqrt(Q/ADV))
    impact_coefficient: float = 0.1  # λ parameter

    # Slippage
    base_slippage_bps: float = 2.0
    volatility_slippage_multiplier: float = 1.5

    # Execution delays
    execution_delay_ms: float = 10.0  # Average execution delay

    # Liquidity constraints
    max_order_book_participation: float = 0.05  # Max % of LOB we can take

    # Market hours (affects holding costs)
    trading_hours_per_day: int = 6  # Crypto: 24, US equities: 6.5


@dataclass
class EconomicMetrics:
    """Economic validation metrics."""

    # Returns
    gross_return: float
    net_return: float
    transaction_cost_drag: float

    # Risk-adjusted
    sharpe_ratio: float
    deflated_sharpe_ratio: float  # Accounting for multiple testing
    sortino_ratio: float
    calmar_ratio: float

    # Drawdown
    max_drawdown: float
    avg_drawdown: float
    drawdown_duration: int

    # Trade statistics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_holding_period: float

    # Economic viability
    breakeven_cost_bps: float  # Max cost where strategy breaks even
    cost_capacity: float  # How much cost can strategy absorb

    # Statistical significance
    t_statistic: float
    p_value: float
    is_significant: bool


class EconomicValidator:
    """
    Validates economic viability of trading strategies.

    Checks:
    1. Net returns after realistic costs
    2. Statistical significance
    3. Market impact constraints
    4. Execution feasibility
    """

    def __init__(self, market_conditions: MarketConditions = None):
        """Initialize validator with market conditions."""
        self.market_conditions = market_conditions or MarketConditions()

    def validate_strategy(
        self,
        predictions_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        volume_df: Optional[pd.DataFrame] = None,
    ) -> EconomicMetrics:
        """
        Comprehensive economic validation.

        Args:
            predictions_df: Model predictions with probabilities
            prices_df: Price data (OHLCV)
            volume_df: Volume data for market impact calculation

        Returns:
            EconomicMetrics object with all validation results
        """
        # Merge data
        df = pd.merge_asof(
            predictions_df.sort_values("timestamp"),
            prices_df.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
        )

        if volume_df is not None:
            df = pd.merge_asof(
                df.sort_values("timestamp"),
                volume_df.sort_values("timestamp"),
                on="timestamp",
                direction="nearest",
            )

        # Simulate realistic trading
        trades = self._simulate_realistic_execution(df)

        # Calculate returns
        gross_return = self._calculate_gross_return(trades)
        net_return = self._calculate_net_return(trades)
        transaction_costs = gross_return - net_return

        # Risk metrics
        returns = self._get_returns_series(trades)
        sharpe = self._calculate_sharpe_ratio(returns)
        deflated_sharpe = self._calculate_deflated_sharpe(returns, n_trials=10)
        sortino = self._calculate_sortino_ratio(returns)

        # Drawdown analysis
        equity_curve = self._build_equity_curve(trades)
        max_dd, avg_dd, dd_duration = self._analyze_drawdown(equity_curve)
        calmar = net_return / max_dd if max_dd > 0 else 0

        # Trade statistics
        win_rate = self._calculate_win_rate(trades)
        profit_factor = self._calculate_profit_factor(trades)
        avg_holding = self._calculate_avg_holding_period(trades)

        # Economic viability
        breakeven_cost = self._calculate_breakeven_cost(trades)
        cost_capacity = breakeven_cost - self.market_conditions.taker_fee_bps

        # Statistical significance
        t_stat, p_value = self._test_significance(returns)
        is_significant = p_value < 0.05

        return EconomicMetrics(
            gross_return=gross_return,
            net_return=net_return,
            transaction_cost_drag=transaction_costs,
            sharpe_ratio=sharpe,
            deflated_sharpe_ratio=deflated_sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            drawdown_duration=dd_duration,
            total_trades=len(trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_holding_period=avg_holding,
            breakeven_cost_bps=breakeven_cost,
            cost_capacity=cost_capacity,
            t_statistic=t_stat,
            p_value=p_value,
            is_significant=is_significant,
        )

    def _simulate_realistic_execution(self, df: pd.DataFrame) -> List[Dict]:
        """Simulate trading with realistic market microstructure."""
        trades = []
        position = None

        for idx, row in df.iterrows():
            timestamp = row["timestamp"]
            prediction = row["prediction"]
            probability = row.get("probability", 0.5)

            # Price data
            mid_price = row.get("mid_price")
            if mid_price is None:
                mid_price = row.get("close")
            if mid_price is None:
                continue

            bid = row.get("bid")
            if bid is None:
                bid = mid_price * 0.9995

            ask = row.get("ask")
            if ask is None:
                ask = mid_price * 1.0005

            spread = ask - bid

            # Volatility (for slippage calculation)
            volatility = row.get("volatility", 0.001)

            # Volume (for market impact)
            volume = row.get("volume", 1000000)

            # Generate signal
            signal = self._signal_from_prediction(prediction, probability)

            # Close existing position
            if position is not None and signal != position["direction"]:
                exit_price, exit_cost = self._calculate_execution_price(
                    -position["direction"],
                    mid_price,
                    bid,
                    ask,
                    spread,
                    volatility,
                    volume,
                )

                # Calculate PnL
                if position["direction"] == 1:  # Long
                    pnl = exit_price - position["entry_price"]
                else:  # Short
                    pnl = position["entry_price"] - exit_price

                position["exit_price"] = exit_price
                position["exit_timestamp"] = timestamp
                position["exit_cost_bps"] = exit_cost
                position["pnl"] = pnl
                position["holding_period"] = (
                    timestamp - position["entry_timestamp"]
                ).total_seconds()

                trades.append(position)
                position = None

            # Open new position
            if signal != 0 and position is None:
                entry_price, entry_cost = self._calculate_execution_price(
                    signal, mid_price, bid, ask, spread, volatility, volume
                )

                position = {
                    "entry_timestamp": timestamp,
                    "direction": signal,
                    "entry_price": entry_price,
                    "entry_cost_bps": entry_cost,
                    "prediction_prob": probability,
                }

        # Close final position
        if position is not None:
            final_row = df.iloc[-1]
            mid_price = final_row.get("mid_price", final_row.get("close"))
            bid = final_row.get("bid", mid_price * 0.9995)
            ask = final_row.get("ask", mid_price * 1.0005)
            spread = ask - bid
            volatility = final_row.get("volatility", 0.001)
            volume = final_row.get("volume", 1000000)

            exit_price, exit_cost = self._calculate_execution_price(
                -position["direction"], mid_price, bid, ask, spread, volatility, volume
            )

            if position["direction"] == 1:
                pnl = exit_price - position["entry_price"]
            else:
                pnl = position["entry_price"] - exit_price

            position["exit_price"] = exit_price
            position["exit_timestamp"] = final_row["timestamp"]
            position["exit_cost_bps"] = exit_cost
            position["pnl"] = pnl
            position["holding_period"] = (
                final_row["timestamp"] - position["entry_timestamp"]
            ).total_seconds()

            trades.append(position)

        return trades

    def _calculate_execution_price(
        self,
        direction: int,
        mid_price: float,
        bid: float,
        ask: float,
        spread: float,
        volatility: float,
        volume: float,
    ) -> Tuple[float, float]:
        """
        Calculate realistic execution price including all costs.

        Returns:
            (execution_price, total_cost_bps)
        """
        # Base execution price
        if direction == 1:  # Buy (taker)
            base_price = ask
            fee_bps = self.market_conditions.taker_fee_bps
        else:  # Sell (taker)
            base_price = bid
            fee_bps = self.market_conditions.taker_fee_bps

        # Slippage (volatility-dependent)
        slippage_bps = self.market_conditions.base_slippage_bps * (
            1 + self.market_conditions.volatility_slippage_multiplier * volatility
        )

        # Market impact (square-root model)
        # Assume we trade 1% of volume
        trade_size = volume * 0.01
        impact_bps = (
            self.market_conditions.impact_coefficient
            * np.sqrt(trade_size / volume)
            * 10000
        )

        # Total cost
        total_cost_bps = fee_bps + slippage_bps + impact_bps

        # Execution price
        if direction == 1:
            execution_price = base_price * (1 + total_cost_bps / 10000)
        else:
            execution_price = base_price * (1 - total_cost_bps / 10000)

        return execution_price, total_cost_bps

    def _signal_from_prediction(self, prediction: int, probability: float) -> int:
        """Convert prediction to trading signal."""
        threshold = 0.52  # Confidence threshold (lowered for demo)

        if probability < threshold:
            return 0

        if prediction == 2:  # Up
            return 1
        elif prediction == 0:  # Down
            return -1
        else:
            return 0

    def _calculate_gross_return(self, trades: List[Dict]) -> float:
        """Calculate gross return (before costs)."""
        if not trades:
            return 0.0

        total_pnl = sum(t["pnl"] for t in trades)
        avg_entry = np.mean([t["entry_price"] for t in trades])

        return (total_pnl / avg_entry) * 100  # Percentage

    def _calculate_net_return(self, trades: List[Dict]) -> float:
        """Calculate net return (after costs)."""
        if not trades:
            return 0.0

        total_pnl = 0
        for t in trades:
            # Subtract costs
            entry_cost = t["entry_price"] * t["entry_cost_bps"] / 10000
            exit_cost = t["exit_price"] * t["exit_cost_bps"] / 10000
            net_pnl = t["pnl"] - entry_cost - exit_cost
            total_pnl += net_pnl

        avg_entry = np.mean([t["entry_price"] for t in trades])
        return (total_pnl / avg_entry) * 100

    def _get_returns_series(self, trades: List[Dict]) -> pd.Series:
        """Get returns series for statistical analysis."""
        if not trades:
            return pd.Series([0])

        returns = []
        for t in trades:
            entry_cost = t["entry_price"] * t["entry_cost_bps"] / 10000
            exit_cost = t["exit_price"] * t["exit_cost_bps"] / 10000
            net_pnl = t["pnl"] - entry_cost - exit_cost
            ret = net_pnl / t["entry_price"]
            returns.append(ret)

        return pd.Series(returns)

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio (annualized)."""
        if len(returns) < 2:
            return 0.0

        # Annualization factor (assume 252 trading days, adjust for holding period)
        periods_per_year = 252 * self.market_conditions.trading_hours_per_day

        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        if std_return == 0:
            return 0.0

        sharpe = np.sqrt(periods_per_year) * mean_return / std_return
        return sharpe

    def _calculate_deflated_sharpe(
        self, returns: pd.Series, n_trials: int = 10
    ) -> float:
        """
        Calculate deflated Sharpe ratio (Bailey & López de Prado, 2014).

        Accounts for multiple testing bias.
        """
        sharpe = self._calculate_sharpe_ratio(returns)
        n = len(returns)

        if n < 2:
            return 0.0

        # Estimate expected maximum Sharpe from n_trials
        # Using the approximation: E[max SR] ≈ (1 - γ) * √(2 * ln(n_trials))
        # where γ is Euler-Mascheroni constant
        gamma = 0.5772
        expected_max_sharpe = (1 - gamma) * np.sqrt(2 * np.log(n_trials))

        # Standard error of Sharpe
        se_sharpe = np.sqrt((1 + sharpe**2 / 2) / n)

        # Deflated Sharpe
        deflated_sharpe = (sharpe - expected_max_sharpe) / se_sharpe

        return deflated_sharpe

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        if len(returns) < 2:
            return 0.0

        periods_per_year = 252 * self.market_conditions.trading_hours_per_day

        mean_return = np.mean(returns)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            return np.inf if mean_return > 0 else 0.0

        downside_std = np.std(downside_returns, ddof=1)

        if downside_std == 0:
            return 0.0

        sortino = np.sqrt(periods_per_year) * mean_return / downside_std
        return sortino

    def _build_equity_curve(self, trades: List[Dict]) -> np.ndarray:
        """Build cumulative equity curve."""
        if not trades:
            return np.array([1.0])

        returns = self._get_returns_series(trades)
        equity = np.cumprod(1 + returns.values)
        equity = np.insert(equity, 0, 1.0)  # Start at 1

        return equity

    def _analyze_drawdown(self, equity_curve: np.ndarray) -> Tuple[float, float, int]:
        """Analyze drawdown characteristics."""
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max

        max_dd = abs(np.min(drawdown))
        avg_dd = abs(np.mean(drawdown[drawdown < 0])) if np.any(drawdown < 0) else 0.0

        # Drawdown duration (longest period underwater)
        underwater = drawdown < -0.01  # More than 1% drawdown
        if np.any(underwater):
            dd_duration = self._longest_consecutive_true(underwater)
        else:
            dd_duration = 0

        return max_dd, avg_dd, dd_duration

    def _longest_consecutive_true(self, arr: np.ndarray) -> int:
        """Find longest consecutive True values."""
        if not np.any(arr):
            return 0

        changes = np.diff(np.concatenate([[False], arr, [False]]).astype(int))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        lengths = ends - starts

        return int(np.max(lengths)) if len(lengths) > 0 else 0

    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate."""
        if not trades:
            return 0.0

        wins = sum(1 for t in trades if t["pnl"] > 0)
        return (wins / len(trades)) * 100

    def _calculate_profit_factor(self, trades: List[Dict]) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if not trades:
            return 0.0

        gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))

        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def _calculate_avg_holding_period(self, trades: List[Dict]) -> float:
        """Calculate average holding period in seconds."""
        if not trades:
            return 0.0

        return np.mean([t["holding_period"] for t in trades])

    def _calculate_breakeven_cost(self, trades: List[Dict]) -> float:
        """Calculate maximum transaction cost where strategy breaks even."""
        if not trades:
            return 0.0

        gross_return = self._calculate_gross_return(trades)
        avg_trades_per_return = len(trades) * 2  # Entry + exit

        # Breakeven cost per trade
        breakeven_bps = (gross_return / avg_trades_per_return) * 10000

        return breakeven_bps

    def _test_significance(self, returns: pd.Series) -> Tuple[float, float]:
        """Test statistical significance of returns."""
        if len(returns) < 2:
            return 0.0, 1.0

        # One-sample t-test (H0: mean return = 0)
        t_stat, p_value = stats.ttest_1samp(returns, 0)

        return t_stat, p_value

    def generate_report(self, metrics: EconomicMetrics) -> str:
        """Generate human-readable validation report."""
        report = []
        report.append("=" * 80)
        report.append("ECONOMIC VALIDATION REPORT")
        report.append("=" * 80)

        report.append("\n[RETURNS]")
        report.append(f"  Gross Return:          {metrics.gross_return:>8.2f}%")
        report.append(f"  Net Return:            {metrics.net_return:>8.2f}%")
        report.append(
            f"  Transaction Cost Drag: {metrics.transaction_cost_drag:>8.2f}%"
        )

        report.append("\n[RISK-ADJUSTED PERFORMANCE]")
        report.append(f"  Sharpe Ratio:          {metrics.sharpe_ratio:>8.2f}")
        report.append(f"  Deflated Sharpe Ratio: {metrics.deflated_sharpe_ratio:>8.2f}")
        report.append(f"  Sortino Ratio:         {metrics.sortino_ratio:>8.2f}")
        report.append(f"  Calmar Ratio:          {metrics.calmar_ratio:>8.2f}")

        report.append("\n[DRAWDOWN ANALYSIS]")
        report.append(f"  Maximum Drawdown:      {metrics.max_drawdown*100:>8.2f}%")
        report.append(f"  Average Drawdown:      {metrics.avg_drawdown*100:>8.2f}%")
        report.append(
            f"  Drawdown Duration:     {metrics.drawdown_duration:>8d} periods"
        )

        report.append("\n[TRADE STATISTICS]")
        report.append(f"  Total Trades:          {metrics.total_trades:>8d}")
        report.append(f"  Win Rate:              {metrics.win_rate:>8.2f}%")
        report.append(f"  Profit Factor:         {metrics.profit_factor:>8.2f}")
        report.append(
            f"  Avg Holding Period:    {metrics.avg_holding_period:>8.0f} seconds"
        )

        report.append("\n[ECONOMIC VIABILITY]")
        report.append(
            f"  Breakeven Cost:        {metrics.breakeven_cost_bps:>8.2f} bps"
        )
        report.append(f"  Cost Capacity:         {metrics.cost_capacity:>8.2f} bps")

        viability = "YES" if metrics.cost_capacity > 0 else "NO"
        report.append(f"  Economically Viable:   {viability:>8s}")

        report.append("\n[STATISTICAL SIGNIFICANCE]")
        report.append(f"  t-statistic:           {metrics.t_statistic:>8.2f}")
        report.append(f"  p-value:               {metrics.p_value:>8.4f}")

        significance = "YES" if metrics.is_significant else "NO"
        report.append(f"  Statistically Sig.:    {significance:>8s} (alpha=0.05)")

        report.append("\n" + "=" * 80)

        # Overall assessment
        if (
            metrics.is_significant
            and metrics.cost_capacity > 0
            and metrics.net_return > 0
        ):
            report.append("VERDICT: Strategy passes economic validation [PASS]")
        else:
            report.append("VERDICT: Strategy FAILS economic validation [FAIL]")

            if not metrics.is_significant:
                report.append("  - Returns not statistically significant")
            if metrics.cost_capacity <= 0:
                report.append("  - Cannot absorb realistic transaction costs")
            if metrics.net_return <= 0:
                report.append("  - Negative net returns")

        report.append("=" * 80)

        return "\n".join(report)


# Demonstration
if __name__ == "__main__":
    print("Economic Validation Framework - Demo")
    print("=" * 80)

    # Generate synthetic strategy results
    np.random.seed(42)
    n = 500

    timestamps = pd.date_range("2024-01-01", periods=n, freq="1min")

    # Synthetic price with trend
    price = 100.0
    prices = []
    volumes = []

    for _ in range(n):
        price += np.random.randn() * 0.1
        prices.append(price)
        volumes.append(np.random.lognormal(14, 1))  # Realistic volume distribution

    prices_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": prices,
            "mid_price": prices,
            "bid": np.array(prices) * 0.9998,
            "ask": np.array(prices) * 1.0002,
            "volatility": np.random.uniform(0.0005, 0.002, n),
            "volume": volumes,
        }
    )

    # Predictions with edge (60% accuracy)
    true_direction = np.diff(prices) > 0
    predictions = []
    probabilities = []

    for i in range(len(true_direction)):
        if np.random.rand() < 0.60:  # 60% accuracy
            pred = 2 if true_direction[i] else 0
            prob = np.random.uniform(0.6, 0.9)
        else:
            pred = 0 if true_direction[i] else 2
            prob = np.random.uniform(0.5, 0.7)

        predictions.append(pred)
        probabilities.append(prob)

    predictions_df = pd.DataFrame(
        {
            "timestamp": timestamps[1:],
            "prediction": predictions,
            "probability": probabilities,
        }
    )

    # Run economic validation
    market_conditions = MarketConditions(
        maker_fee_bps=1.0,
        taker_fee_bps=5.0,
        impact_coefficient=0.1,
        base_slippage_bps=2.0,
    )

    validator = EconomicValidator(market_conditions)
    metrics = validator.validate_strategy(predictions_df, prices_df, prices_df)

    # Generate report
    report = validator.generate_report(metrics)
    print(report)
