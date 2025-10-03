"""
Advanced Hyperparameter Tuning with Optuna

Automated optimization for:
- LSTM architecture
- Transformer models
- Ensemble weights
- Trading strategy parameters
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import json
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.lstm_model import OrderBookLSTM, AttentionLSTM
from models.transformer_model import OrderBookTransformer
from models.train_lstm import Trainer, FeatureScaler


class HyperparameterOptimizer:
    """Optimize model hyperparameters using Optuna."""

    def __init__(self, features_df: pd.DataFrame, n_trials: int = 50):
        """
        Initialize optimizer.

        Args:
            features_df: DataFrame with features and labels
            n_trials: Number of optimization trials
        """
        self.features_df = features_df
        self.n_trials = n_trials
        self.best_params = {}
        self.study = None

    def optimize_lstm(self, device: str = "cpu") -> Dict:
        """Optimize LSTM hyperparameters."""

        def objective(trial):
            # Suggest hyperparameters
            hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
            num_layers = trial.suggest_int("num_layers", 1, 4)
            dropout = trial.suggest_float("dropout", 0.1, 0.5)
            learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
            bidirectional = trial.suggest_categorical("bidirectional", [True, False])

            # Create model
            input_size = len(
                [c for c in self.features_df.columns if c not in ["label", "timestamp"]]
            )

            model = OrderBookLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_classes=3,
                dropout=dropout,
                bidirectional=bidirectional,
            ).to(device)

            # Train
            trainer = Trainer(device=device)
            train_loader, val_loader, feature_names = trainer.prepare_data(
                self.features_df,
                sequence_length=20,
                train_split=0.8,
                batch_size=batch_size,
            )

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # Train for limited epochs (early stopping)
            best_val_acc = 0
            patience = 5
            patience_counter = 0

            for epoch in range(20):
                # Training
                train_loss, train_acc = trainer.train_epoch(
                    model, train_loader, criterion, optimizer
                )

                # Validation
                val_loss, val_acc = trainer.validate(model, val_loader, criterion)

                # Report intermediate value for pruning
                trial.report(val_acc, epoch)

                # Handle pruning
                if trial.should_prune():
                    raise optuna.TrialPruned()

                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            return best_val_acc

        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )

        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        self.study = study
        self.best_params = study.best_params

        print(f"\n[OK] Best LSTM hyperparameters found:")
        print(f"     Validation Accuracy: {study.best_value:.4f}")
        print(f"     Parameters: {study.best_params}")

        return study.best_params

    def optimize_transformer(self, device: str = "cpu") -> Dict:
        """Optimize Transformer hyperparameters."""

        def objective(trial):
            # Suggest hyperparameters
            d_model = trial.suggest_categorical("d_model", [64, 128, 256])
            nhead = trial.suggest_categorical("nhead", [4, 8, 16])
            num_layers = trial.suggest_int("num_encoder_layers", 2, 6)
            dim_feedforward = trial.suggest_categorical(
                "dim_feedforward", [256, 512, 1024]
            )
            dropout = trial.suggest_float("dropout", 0.1, 0.3)
            learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

            # Create model
            input_size = len(
                [c for c in self.features_df.columns if c not in ["label", "timestamp"]]
            )

            model = OrderBookTransformer(
                input_size=input_size,
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            ).to(device)

            # Train
            trainer = Trainer(device=device)
            train_loader, val_loader, feature_names = trainer.prepare_data(
                self.features_df,
                sequence_length=20,
                train_split=0.8,
                batch_size=batch_size,
            )

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # Train
            best_val_acc = 0
            patience = 5
            patience_counter = 0

            for epoch in range(15):
                train_loss, train_acc = trainer.train_epoch(
                    model, train_loader, criterion, optimizer
                )

                val_loss, val_acc = trainer.validate(model, val_loader, criterion)

                trial.report(val_acc, epoch)

                if trial.should_prune():
                    raise optuna.TrialPruned()

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            return best_val_acc

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=5),
        )

        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        self.study = study
        self.best_params = study.best_params

        print(f"\n[OK] Best Transformer hyperparameters found:")
        print(f"     Validation Accuracy: {study.best_value:.4f}")
        print(f"     Parameters: {study.best_params}")

        return study.best_params

    def optimize_trading_strategy(self) -> Dict:
        """Optimize trading strategy parameters."""

        def objective(trial):
            # Strategy parameters
            confidence_threshold = trial.suggest_float(
                "confidence_threshold", 0.5, 0.95
            )
            position_size_pct = trial.suggest_float("position_size_pct", 0.01, 0.1)
            stop_loss_pct = trial.suggest_float("stop_loss_pct", 0.01, 0.05)
            take_profit_pct = trial.suggest_float("take_profit_pct", 0.02, 0.1)

            # Simplified backtesting simulation
            # (In production, use full backtesting engine)
            np.random.seed(trial.number)

            # Simulate trades
            n_trades = 100
            wins = 0
            total_pnl = 0

            for _ in range(n_trades):
                # Random signal confidence
                confidence = np.random.uniform(0.4, 1.0)

                if confidence < confidence_threshold:
                    continue

                # Simulate trade outcome
                win_prob = confidence  # Higher confidence = higher win rate
                is_win = np.random.rand() < win_prob

                if is_win:
                    pnl = position_size_pct * take_profit_pct
                    wins += 1
                else:
                    pnl = -position_size_pct * stop_loss_pct

                total_pnl += pnl

            # Objective: maximize Sharpe-like ratio
            if n_trades > 0:
                avg_pnl = total_pnl / n_trades
                std_pnl = 0.01  # Simplified
                sharpe = avg_pnl / (std_pnl + 1e-10)
            else:
                sharpe = 0

            return sharpe

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        print(f"\n[OK] Best trading strategy parameters:")
        print(f"     Sharpe Ratio: {study.best_value:.4f}")
        print(f"     Parameters: {study.best_params}")

        return study.best_params

    def save_results(self, filepath: str):
        """Save optimization results."""
        if self.study is None:
            print("No study to save. Run optimization first.")
            return

        results = {
            "timestamp": datetime.now().isoformat(),
            "best_params": self.best_params,
            "best_value": self.study.best_value,
            "n_trials": len(self.study.trials),
            "trials": [],
        }

        for trial in self.study.trials:
            results["trials"].append(
                {
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "state": trial.state.name,
                }
            )

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        print(f"[OK] Results saved to {filepath}")

    def plot_optimization_history(self):
        """Plot optimization history."""
        if self.study is None:
            print("No study to plot. Run optimization first.")
            return

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Optimization history
        values = [trial.value for trial in self.study.trials if trial.value is not None]
        axes[0].plot(values, marker="o", alpha=0.7)
        axes[0].set_xlabel("Trial Number")
        axes[0].set_ylabel("Validation Accuracy")
        axes[0].set_title("Optimization History")
        axes[0].grid(True, alpha=0.3)

        # Best value line
        best_values = []
        best_so_far = -np.inf
        for v in values:
            best_so_far = max(best_so_far, v)
            best_values.append(best_so_far)
        axes[0].plot(best_values, "r--", label="Best So Far", linewidth=2)
        axes[0].legend()

        # Parameter importance (if more than 5 trials)
        if len(self.study.trials) > 5:
            try:
                importance = optuna.importance.get_param_importances(self.study)
                params = list(importance.keys())
                importances = list(importance.values())

                axes[1].barh(params, importances)
                axes[1].set_xlabel("Importance")
                axes[1].set_title("Hyperparameter Importance")
                axes[1].grid(True, alpha=0.3)
            except:
                axes[1].text(
                    0.5,
                    0.5,
                    "Not enough trials\nfor importance analysis",
                    ha="center",
                    va="center",
                    fontsize=12,
                )

        plt.tight_layout()
        return fig


# Demo
if __name__ == "__main__":
    print("=" * 80)
    print("HYPERPARAMETER OPTIMIZATION DEMO")
    print("=" * 80)

    # Generate synthetic data
    print("\n1. Generating synthetic training data...")
    np.random.seed(42)

    n_samples = 1000
    n_features = 20

    features = np.random.randn(n_samples, n_features)
    labels = np.random.randint(0, 3, n_samples)

    df = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(n_features)])
    df["label"] = labels

    print(f"   Generated {n_samples} samples with {n_features} features")

    # Optimize LSTM (fast demo with 10 trials)
    print("\n2. Optimizing LSTM hyperparameters (10 trials)...")
    optimizer = HyperparameterOptimizer(df, n_trials=10)

    best_lstm_params = optimizer.optimize_lstm(device="cpu")

    # Save results
    print("\n3. Saving optimization results...")
    os.makedirs("models/tuning_results", exist_ok=True)
    optimizer.save_results("models/tuning_results/lstm_optimization.json")

    # Plot
    print("\n4. Generating visualization...")
    fig = optimizer.plot_optimization_history()
    if fig:
        fig.savefig("models/tuning_results/optimization_history.png", dpi=150)
        print("   [OK] Saved to models/tuning_results/optimization_history.png")

    print("\n" + "=" * 80)
    print("Hyperparameter optimization complete!")
    print("=" * 80)
