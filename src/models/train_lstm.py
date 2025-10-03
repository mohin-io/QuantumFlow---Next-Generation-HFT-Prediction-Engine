"""
Training Pipeline for LSTM Order Book Forecasting

Complete training script with:
- Data loading and preprocessing
- Model training with validation
- Checkpoint saving
- TensorBoard logging
- Early stopping
"""

import os
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yaml

import sys

sys.path.append("..")

from models.lstm_model import OrderBookLSTM, AttentionLSTM, count_parameters
from features.feature_pipeline import (
    FeaturePipeline,
    FeaturePipelineConfig,
    create_training_dataset,
)


class OrderBookDataset(Dataset):
    """PyTorch Dataset for order book sequences."""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Trainer:
    """Training manager for order book forecasting models."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create directories
        self.checkpoint_dir = Path(config["checkpointing"]["save_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(config["logging"]["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer
        self.writer = SummaryWriter(
            log_dir=self.log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def prepare_data(self, features_df):
        """Prepare training, validation, and test sets."""
        print("Preparing training data...")

        # Create dataset
        dataset = create_training_dataset(
            features_df,
            prediction_horizon=self.config["training"]["prediction_horizon"],
            threshold_bps=self.config["model"]["output"]["threshold_bps"],
            sequence_length=self.config["training"]["sequence_length"],
        )

        X = dataset["X"]
        y = dataset["y"]

        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=self.config["training"]["test_split"],
            random_state=42,
            stratify=y,
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=self.config["training"]["val_split"]
            / (1 - self.config["training"]["test_split"]),
            random_state=42,
            stratify=y_temp,
        )

        # Normalize
        scaler = StandardScaler()
        n_samples, seq_len, n_features = X_train.shape

        X_train_reshaped = X_train.reshape(-1, n_features)
        X_val_reshaped = X_val.reshape(-1, n_features)
        X_test_reshaped = X_test.reshape(-1, n_features)

        X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(
            n_samples, seq_len, n_features
        )
        X_val_scaled = scaler.transform(X_val_reshaped).reshape(
            len(X_val), seq_len, n_features
        )
        X_test_scaled = scaler.transform(X_test_reshaped).reshape(
            len(X_test), seq_len, n_features
        )

        # Create datasets
        train_dataset = OrderBookDataset(X_train_scaled, y_train)
        val_dataset = OrderBookDataset(X_val_scaled, y_val)
        test_dataset = OrderBookDataset(X_test_scaled, y_test)

        # Create loaders
        batch_size = self.config["training"]["batch_size"]

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(
            f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
        )

        return train_loader, val_loader, test_loader, scaler, dataset["feature_names"]

    def create_model(self, input_size):
        """Create model based on config."""
        model_config = self.config["model"]["architecture"]

        model = OrderBookLSTM(
            input_size=input_size,
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
            num_classes=self.config["model"]["output"]["num_classes"],
            dropout=model_config["dropout"],
            bidirectional=model_config["bidirectional"],
        ).to(self.device)

        print(f"\nModel created: {count_parameters(model):,} parameters")

        return model

    def train_epoch(self, model, train_loader, criterion, optimizer):
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

            optimizer.zero_grad()
            logits, _ = model(batch_X)
            loss = criterion(logits, batch_y)

            loss.backward()

            # Gradient clipping
            if self.config["training"]["gradient_clip"]["enabled"]:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=self.config["training"]["gradient_clip"]["max_norm"],
                )

            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

        return total_loss / len(train_loader), correct / total

    def validate(self, model, val_loader, criterion):
        """Validate model."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                logits, _ = model(batch_X)
                loss = criterion(logits, batch_y)

                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

        return total_loss / len(val_loader), correct / total

    def train(self, train_loader, val_loader, model, feature_names):
        """Complete training loop."""
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.config["training"]["lr_scheduler"]["factor"],
            patience=self.config["training"]["lr_scheduler"]["patience"],
            min_lr=self.config["training"]["lr_scheduler"]["min_lr"],
        )

        num_epochs = self.config["training"]["epochs"]

        print(f"\nStarting training for {num_epochs} epochs...")
        print("=" * 80)

        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(
                model, train_loader, criterion, optimizer
            )

            # Validate
            val_loss, val_acc = self.validate(model, val_loader, criterion)

            # Update scheduler
            scheduler.step(val_loss)

            # Log to TensorBoard
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/val", val_acc, epoch)
            self.writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

            # Print progress
            if (epoch + 1) % self.config["logging"]["log_every_n_steps"] == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}]")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
                print()

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "config": self.config,
                    "feature_names": feature_names,
                }

                torch.save(checkpoint, self.checkpoint_dir / "best_model.pth")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.config["training"]["early_stopping"]["enabled"]:
                if (
                    self.patience_counter
                    >= self.config["training"]["early_stopping"]["patience"]
                ):
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        print("=" * 80)
        print("Training complete!")

        # Save final model
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "config": self.config,
            "feature_names": feature_names,
        }

        torch.save(checkpoint, self.checkpoint_dir / "final_model.pth")

        self.writer.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train LSTM for Order Book Forecasting"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/models/lstm_config.yaml",
        help="Path to config file",
    )
    parser.add_argument("--data", type=str, help="Path to features CSV (optional)")

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("LSTM TRAINING PIPELINE")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Initialize trainer
    trainer = Trainer(config)

    # Load or generate data
    if args.data:
        features_df = pd.read_csv(args.data)
    else:
        print("\nGenerating synthetic data for demonstration...")
        # Generate demo data
        from features.feature_pipeline import FeaturePipeline, FeaturePipelineConfig

        # Simple synthetic order book
        snapshots = []
        mid_price = 50000.0
        for i in range(2000):
            mid_price += np.random.normal(0, 10)
            bids = [
                [mid_price - (j + 1) * 0.5, 50 + np.random.normal(0, 10)]
                for j in range(20)
            ]
            asks = [
                [mid_price + (j + 1) * 0.5, 50 + np.random.normal(0, 10)]
                for j in range(20)
            ]
            snapshots.append({"timestamp": i, "bids": bids, "asks": asks})

        df = pd.DataFrame(snapshots)

        pipeline_config = FeaturePipelineConfig(
            ofi_levels=[1, 5], ofi_windows=[10], volatility_windows=[], ohlc_bar_size=10
        )
        pipeline = FeaturePipeline(pipeline_config)
        features_df = pipeline.compute_all_features(df, include_volatility=False)

    # Prepare data
    train_loader, val_loader, test_loader, scaler, feature_names = trainer.prepare_data(
        features_df
    )

    # Create model
    input_size = len(feature_names)
    model = trainer.create_model(input_size)

    # Train
    trainer.train(train_loader, val_loader, model, feature_names)

    print(f"\nCheckpoints saved to: {trainer.checkpoint_dir}")
    print(f"TensorBoard logs: {trainer.log_dir}")
    print("\nTo view training curves: tensorboard --logdir", trainer.log_dir)


if __name__ == "__main__":
    main()
