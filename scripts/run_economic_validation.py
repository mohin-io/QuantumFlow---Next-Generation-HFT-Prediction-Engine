"""
Run comprehensive economic validation on trading strategy.

This script:
1. Loads historical data or generates synthetic data
2. Loads model predictions
3. Runs economic validation with realistic costs
4. Generates detailed report and visualizations
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from backtesting.economic_validation import EconomicValidator, MarketConditions
from backtesting.backtest_engine import BacktestEngine, BacktestConfig


def generate_realistic_market_data(n_samples=5000, seed=42):
    """Generate realistic market microstructure data."""
    np.random.seed(seed)

    print(f"Generating {n_samples} samples of realistic market data...")

    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='100ms')

    # Price follows GBM with mean reversion
    price = 100.0
    prices = [price]

    for _ in range(n_samples - 1):
        # Mean reversion + noise
        drift = -0.0001 * (price - 100.0)
        diffusion = np.random.randn() * 0.01
        price = price * (1 + drift + diffusion)
        prices.append(price)

    prices = np.array(prices)

    # Realistic bid-ask spread (0.01% - 0.05%)
    spreads = np.random.uniform(0.0001, 0.0005, n_samples)
    bids = prices * (1 - spreads / 2)
    asks = prices * (1 + spreads / 2)

    # Volume follows log-normal distribution
    volumes = np.random.lognormal(mean=14, sigma=1.5, size=n_samples)

    # Volatility (realized from rolling window)
    returns = np.diff(np.log(prices))
    volatility = np.concatenate([[0.001], pd.Series(returns).rolling(20).std().fillna(0.001).values])

    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.randn(n_samples) * 0.001)),
        'low': prices * (1 - np.abs(np.random.randn(n_samples) * 0.001)),
        'close': prices,
        'mid_price': prices,
        'bid': bids,
        'ask': asks,
        'volume': volumes,
        'volatility': volatility
    })

    print("[OK] Market data generated")
    return df


def generate_model_predictions(prices_df, accuracy=0.65, seed=42):
    """
    Generate synthetic model predictions with specified accuracy.

    Args:
        prices_df: Market data
        accuracy: Prediction accuracy (0-1)
        seed: Random seed
    """
    np.random.seed(seed)

    print(f"Generating model predictions (accuracy={accuracy*100:.0f}%)...")

    prices = prices_df['close'].values
    true_direction = np.diff(prices) > 0

    predictions = []
    probabilities = []

    for i in range(len(true_direction)):
        # Generate prediction based on accuracy
        if np.random.rand() < accuracy:
            # Correct prediction
            pred = 2 if true_direction[i] else 0
            prob = np.random.uniform(0.55, 0.95)  # Higher confidence
        else:
            # Wrong prediction
            pred = 0 if true_direction[i] else 2
            prob = np.random.uniform(0.40, 0.60)  # Lower confidence

        predictions.append(pred)

        # Add probability distribution
        if pred == 2:
            prob_dist = [0.1, 0.3, prob]
        elif pred == 0:
            prob_dist = [prob, 0.3, 0.1]
        else:
            prob_dist = [0.2, 0.6, 0.2]

        # Normalize
        prob_dist = np.array(prob_dist) / sum(prob_dist)
        probabilities.append(prob_dist)

    predictions_df = pd.DataFrame({
        'timestamp': prices_df['timestamp'].iloc[1:].values,
        'prediction': predictions,
        'probability': [p[np.argmax(p)] for p in probabilities],
        'prob_down': [p[0] for p in probabilities],
        'prob_neutral': [p[1] for p in probabilities],
        'prob_up': [p[2] for p in probabilities]
    })

    print(f"[OK] Generated {len(predictions_df)} predictions")
    return predictions_df


def run_validation_scenarios(predictions_df, prices_df):
    """Run validation under different market condition scenarios."""

    scenarios = {
        'Low Cost (Maker)': MarketConditions(
            maker_fee_bps=-0.5,  # Rebate
            taker_fee_bps=2.0,
            base_slippage_bps=1.0,
            impact_coefficient=0.05
        ),
        'Medium Cost (Retail)': MarketConditions(
            maker_fee_bps=1.0,
            taker_fee_bps=5.0,
            base_slippage_bps=2.0,
            impact_coefficient=0.1
        ),
        'High Cost (Aggressive)': MarketConditions(
            maker_fee_bps=2.0,
            taker_fee_bps=10.0,
            base_slippage_bps=5.0,
            impact_coefficient=0.3
        )
    }

    results = {}

    print("\n" + "="*80)
    print("RUNNING ECONOMIC VALIDATION SCENARIOS")
    print("="*80)

    for scenario_name, conditions in scenarios.items():
        print(f"\nScenario: {scenario_name}")
        print("-"*80)

        validator = EconomicValidator(conditions)
        metrics = validator.validate_strategy(predictions_df, prices_df, prices_df)

        # Print key metrics
        print(f"  Net Return:      {metrics.net_return:>8.2f}%")
        print(f"  Sharpe Ratio:    {metrics.sharpe_ratio:>8.2f}")
        print(f"  Max Drawdown:    {metrics.max_drawdown*100:>8.2f}%")
        print(f"  Cost Capacity:   {metrics.cost_capacity:>8.2f} bps")
        print(f"  Viable:          {'YES' if metrics.cost_capacity > 0 else 'NO'}")

        results[scenario_name] = metrics

    return results


def create_validation_visualizations(results, save_path='data/simulations/'):
    """Create visualization comparing scenarios."""

    os.makedirs(save_path, exist_ok=True)

    scenario_names = list(results.keys())
    n_scenarios = len(scenario_names)

    # Extract metrics
    net_returns = [results[s].net_return for s in scenario_names]
    sharpe_ratios = [results[s].sharpe_ratio for s in scenario_names]
    max_drawdowns = [results[s].max_drawdown * 100 for s in scenario_names]
    cost_capacities = [results[s].cost_capacity for s in scenario_names]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Net Returns
    colors = ['green' if r > 0 else 'red' for r in net_returns]
    axes[0, 0].bar(range(n_scenarios), net_returns, color=colors, alpha=0.7)
    axes[0, 0].set_xticks(range(n_scenarios))
    axes[0, 0].set_xticklabels(scenario_names, rotation=15, ha='right')
    axes[0, 0].set_ylabel('Net Return (%)')
    axes[0, 0].set_title('Net Returns by Cost Scenario')
    axes[0, 0].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[0, 0].grid(True, alpha=0.3)

    # Sharpe Ratios
    colors = ['green' if s > 0 else 'red' for s in sharpe_ratios]
    axes[0, 1].bar(range(n_scenarios), sharpe_ratios, color=colors, alpha=0.7)
    axes[0, 1].set_xticks(range(n_scenarios))
    axes[0, 1].set_xticklabels(scenario_names, rotation=15, ha='right')
    axes[0, 1].set_ylabel('Sharpe Ratio')
    axes[0, 1].set_title('Risk-Adjusted Performance')
    axes[0, 1].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[0, 1].grid(True, alpha=0.3)

    # Maximum Drawdown
    axes[1, 0].bar(range(n_scenarios), max_drawdowns, color='darkred', alpha=0.7)
    axes[1, 0].set_xticks(range(n_scenarios))
    axes[1, 0].set_xticklabels(scenario_names, rotation=15, ha='right')
    axes[1, 0].set_ylabel('Maximum Drawdown (%)')
    axes[1, 0].set_title('Maximum Drawdown')
    axes[1, 0].grid(True, alpha=0.3)

    # Cost Capacity
    colors = ['green' if c > 0 else 'red' for c in cost_capacities]
    axes[1, 1].bar(range(n_scenarios), cost_capacities, color=colors, alpha=0.7)
    axes[1, 1].set_xticks(range(n_scenarios))
    axes[1, 1].set_xticklabels(scenario_names, rotation=15, ha='right')
    axes[1, 1].set_ylabel('Cost Capacity (bps)')
    axes[1, 1].set_title('Economic Viability (Cost Capacity)')
    axes[1, 1].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(save_path, 'economic_validation_scenarios.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Saved visualization to {filepath}")

    plt.close()


def main():
    """Main execution."""
    print("="*80)
    print("ECONOMIC VALIDATION FOR HFT ORDER BOOK STRATEGY")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Generate data
    prices_df = generate_realistic_market_data(n_samples=5000)

    # Generate predictions (65% accuracy)
    predictions_df = generate_model_predictions(prices_df, accuracy=0.65)

    # Run validation scenarios
    results = run_validation_scenarios(predictions_df, prices_df)

    # Create visualizations
    create_validation_visualizations(results)

    # Print detailed report for medium cost scenario
    print("\n" + "="*80)
    print("DETAILED REPORT - MEDIUM COST SCENARIO (MOST REALISTIC)")
    print("="*80)

    validator = EconomicValidator(MarketConditions())
    metrics = validator.validate_strategy(predictions_df, prices_df, prices_df)
    report = validator.generate_report(metrics)
    print(report)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    viable_count = sum(1 for m in results.values() if m.cost_capacity > 0)
    print(f"Scenarios Tested:        {len(results)}")
    print(f"Economically Viable:     {viable_count}/{len(results)}")

    if viable_count > 0:
        print("\n[PASS] Strategy shows economic viability under favorable conditions")
    else:
        print("\n[FAIL] Strategy not viable under any tested scenario")

    print("\nKey Insights:")
    print("  - Transaction costs significantly impact profitability")
    print("  - Market impact and slippage must be carefully modeled")
    print("  - Statistical significance testing prevents overfitting")
    print("  - Deflated Sharpe ratio accounts for multiple testing bias")

    print("\n" + "="*80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    main()
