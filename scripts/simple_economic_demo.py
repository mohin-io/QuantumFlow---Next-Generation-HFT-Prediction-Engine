"""
Simplified economic validation demonstration.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from datetime import datetime

from backtesting.backtest_engine import BacktestEngine, BacktestConfig


def main():
    """Run simple backtest with realistic costs."""
    print("="*80)
    print("SIMPLE ECONOMIC VALIDATION DEMO")
    print("="*80)

    np.random.seed(42)
    n = 2000

    # Generate price data with trend
    timestamps = pd.date_range('2024-01-01', periods=n, freq='1s')
    price = 100.0
    prices = []

    for _ in range(n):
        price += np.random.randn() * 0.05
        prices.append(price)

    prices = np.array(prices)

    prices_df = pd.DataFrame({
        'timestamp': timestamps,
        'mid_price': prices,
        'bid': prices * 0.9998,
        'ask': prices * 1.0002
    })

    # Generate predictions with 65% accuracy
    true_direction = np.diff(prices) > 0
    predictions = []
    probabilities = []

    for i in range(len(true_direction)):
        if np.random.rand() < 0.65:
            pred = 2 if true_direction[i] else 0
            prob = 0.70
        else:
            pred = 0 if true_direction[i] else 2
            prob = 0.60

        predictions.append(pred)
        probabilities.append(prob)

    predictions_df = pd.DataFrame({
        'timestamp': timestamps[1:],
        'prediction': predictions,
        'probability': probabilities
    })

    print(f"\nGenerated {len(predictions_df)} predictions")
    print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")

    # Run backtest with realistic costs
    config = BacktestConfig(
        initial_capital=100000,
        position_size=0.05,  # 5% per trade
        transaction_cost_bps=5.0,  # 0.05% per trade
        confidence_threshold=0.55  # Trade when prob > 55%
    )

    engine = BacktestEngine(config)
    metrics = engine.run(predictions_df, prices_df)

    # Display results
    print("\n" + "="*80)
    print("BACKTEST RESULTS")
    print("="*80)
    print(f"\nTotal Trades:      {metrics['total_trades']:>8d}")
    print(f"Win Rate:          {metrics['win_rate']:>8.2f}%")
    print(f"Total Return:      {metrics['total_return']:>8.2f}%")
    print(f"Sharpe Ratio:      {metrics['sharpe_ratio']:>8.2f}")
    print(f"Max Drawdown:      {metrics['max_drawdown']:>8.2f}%")
    print(f"Profit Factor:     {metrics['profit_factor']:>8.2f}")
    print(f"\nFinal Capital:     ${metrics['final_capital']:>8,.2f}")
    print(f"Total PnL:         ${metrics['total_pnl']:>8,.2f}")

    # Economic assessment
    print("\n" + "="*80)
    print("ECONOMIC ASSESSMENT")
    print("="*80)

    if metrics['total_return'] > 0 and metrics['sharpe_ratio'] > 1.0:
        print("[PASS] Strategy shows positive risk-adjusted returns")
    elif metrics['total_return'] > 0:
        print("[PARTIAL] Strategy profitable but low Sharpe ratio")
    else:
        print("[FAIL] Strategy unprofitable after transaction costs")

    print("\nKey Insights:")
    print(f"  - Transaction costs: 5 bps per trade ({metrics['total_trades']*2} round trips)")
    print(f"  - Average trade PnL: ${metrics['avg_trade_pnl']:.2f}")
    print(f"  - Breakeven accuracy needed: ~55-60% with these costs")
    print(f"  - Actual win rate achieved: {metrics['win_rate']:.1f}%")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
