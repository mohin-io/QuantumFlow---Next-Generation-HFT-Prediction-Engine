"""
Ensemble Meta-learner for Order Book Forecasting

Combines multiple models with different architectures and time horizons:
- LSTM/GRU sequence models
- Transformer attention models
- Bayesian online learners

Ensemble strategies:
- Weighted averaging with dynamic weights
- Stacking with LightGBM/XGBoost
- Performance-based weight updates

References:
- Zhou, Z. H. (2012). Ensemble Methods: Foundations and Algorithms
- Wolpert, D. H. (1992). Stacked generalization
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from collections import deque
import joblib


class WeightedEnsemble:
    """
    Ensemble that combines predictions using weighted averaging.

    Weights are updated dynamically based on recent performance.
    """

    def __init__(self, models: List, initial_weights: Optional[np.ndarray] = None):
        """
        Initialize weighted ensemble.

        Args:
            models: List of model instances
            initial_weights: Initial weights (default: uniform)
        """
        self.models = models
        self.n_models = len(models)

        if initial_weights is None:
            self.weights = np.ones(self.n_models) / self.n_models
        else:
            assert len(initial_weights) == self.n_models
            self.weights = initial_weights / np.sum(initial_weights)

        # Track recent performance
        self.performance_history = deque(maxlen=100)
        self.model_predictions = deque(maxlen=100)
        self.model_actuals = deque(maxlen=100)

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """
        Get ensemble probability predictions.

        Args:
            x: Input features

        Returns:
            Weighted average of model predictions
        """
        predictions = []

        for model in self.models:
            if hasattr(model, "predict_proba"):
                # Bayesian or sklearn-like models
                if isinstance(x, torch.Tensor):
                    x_np = x.cpu().numpy()
                    pred = model.predict_proba(x_np)
                else:
                    pred = model.predict_proba(x)
            else:
                # PyTorch models
                with torch.no_grad():
                    logits = model(x)
                    pred = torch.softmax(logits, dim=-1).cpu().numpy()

            predictions.append(pred)

        # Weighted average
        predictions = np.array(predictions)  # (n_models, batch_size, n_classes)
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)

        return ensemble_pred

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(x)
        return np.argmax(probs, axis=-1)

    def update_weights(self, x: torch.Tensor, y_true: np.ndarray, method="accuracy"):
        """
        Update model weights based on recent performance.

        Args:
            x: Input features
            y_true: True labels
            method: Weight update method ('accuracy', 'loss', 'f1')
        """
        # Get predictions from each model
        model_preds = []
        for model in self.models:
            pred = self.predict_single_model(model, x)
            model_preds.append(pred)

        model_preds = np.array(model_preds)  # (n_models, batch_size)

        # Calculate performance for each model
        if method == "accuracy":
            scores = np.array([np.mean(preds == y_true) for preds in model_preds])
        elif method == "loss":
            # Negative log-likelihood (lower is better)
            scores = np.array(
                [
                    -self._nll(self.predict_proba_single(model, x), y_true)
                    for model in self.models
                ]
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Update weights using softmax with temperature
        temperature = 0.5
        exp_scores = np.exp(scores / temperature)
        self.weights = exp_scores / np.sum(exp_scores)

        # Store history
        self.performance_history.append(scores)
        self.model_predictions.append(model_preds)
        self.model_actuals.append(y_true)

    def predict_single_model(self, model, x):
        """Get prediction from single model."""
        if hasattr(model, "predict"):
            if isinstance(x, torch.Tensor):
                x_np = x.cpu().numpy()
                return model.predict(x_np)
            return model.predict(x)
        else:
            with torch.no_grad():
                logits = model(x)
                return torch.argmax(logits, dim=-1).cpu().numpy()

    def predict_proba_single(self, model, x):
        """Get probabilities from single model."""
        if hasattr(model, "predict_proba"):
            if isinstance(x, torch.Tensor):
                x_np = x.cpu().numpy()
                return model.predict_proba(x_np)
            return model.predict_proba(x)
        else:
            with torch.no_grad():
                logits = model(x)
                return torch.softmax(logits, dim=-1).cpu().numpy()

    @staticmethod
    def _nll(probs, y_true):
        """Negative log-likelihood."""
        n = len(y_true)
        nll = -np.sum(np.log(probs[np.arange(n), y_true] + 1e-10)) / n
        return nll

    def get_weight_history(self) -> np.ndarray:
        """Get historical weights over time."""
        if len(self.performance_history) == 0:
            return None
        return np.array(list(self.performance_history))

    def save(self, filepath: str):
        """Save ensemble configuration."""
        config = {"weights": self.weights, "n_models": self.n_models}
        joblib.dump(config, filepath)

    def load(self, filepath: str):
        """Load ensemble configuration."""
        config = joblib.load(filepath)
        self.weights = config["weights"]


class StackingEnsemble:
    """
    Stacking ensemble using meta-learner.

    Level 0: Base models (LSTM, Transformer, Bayesian)
    Level 1: Meta-model (LightGBM or XGBoost)
    """

    def __init__(self, base_models: List, meta_model=None):
        """
        Initialize stacking ensemble.

        Args:
            base_models: List of base model instances
            meta_model: Meta-learner (if None, uses LightGBM)
        """
        self.base_models = base_models
        self.n_models = len(base_models)

        if meta_model is None:
            try:
                import lightgbm as lgb

                self.meta_model = lgb.LGBMClassifier(
                    n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42
                )
            except ImportError:
                from sklearn.ensemble import RandomForestClassifier

                self.meta_model = RandomForestClassifier(
                    n_estimators=100, max_depth=5, random_state=42
                )
        else:
            self.meta_model = meta_model

        self.is_fitted = False

    def _get_base_predictions(self, x: torch.Tensor) -> np.ndarray:
        """Get predictions from all base models."""
        predictions = []

        for model in self.base_models:
            if hasattr(model, "predict_proba"):
                if isinstance(x, torch.Tensor):
                    x_np = x.cpu().numpy()
                    pred = model.predict_proba(x_np)
                else:
                    pred = model.predict_proba(x)
            else:
                with torch.no_grad():
                    logits = model(x)
                    pred = torch.softmax(logits, dim=-1).cpu().numpy()

            predictions.append(pred)

        # Concatenate all predictions: (batch_size, n_models * n_classes)
        predictions = np.concatenate(predictions, axis=-1)
        return predictions

    def fit(self, x_train: torch.Tensor, y_train: np.ndarray):
        """
        Fit meta-model on base model predictions.

        Args:
            x_train: Training features
            y_train: Training labels
        """
        # Get base model predictions
        meta_features = self._get_base_predictions(x_train)

        # Fit meta-model
        self.meta_model.fit(meta_features, y_train)
        self.is_fitted = True

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """Predict probabilities using stacked ensemble."""
        if not self.is_fitted:
            raise ValueError("Meta-model not fitted. Call fit() first.")

        meta_features = self._get_base_predictions(x)
        return self.meta_model.predict_proba(meta_features)

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(x)
        return np.argmax(probs, axis=-1)

    def save(self, filepath: str):
        """Save ensemble and meta-model."""
        joblib.dump(self.meta_model, filepath)

    def load(self, filepath: str):
        """Load meta-model."""
        self.meta_model = joblib.load(filepath)
        self.is_fitted = True


class MultiHorizonEnsemble:
    """
    Ensemble that trains models on different prediction horizons
    and combines them adaptively.
    """

    def __init__(self, model_class, horizons: List[int], **model_kwargs):
        """
        Initialize multi-horizon ensemble.

        Args:
            model_class: Model class to instantiate
            horizons: List of prediction horizons in ticks
            **model_kwargs: Arguments passed to model constructor
        """
        self.horizons = horizons
        self.models = [model_class(**model_kwargs) for _ in horizons]
        self.weights = np.ones(len(horizons)) / len(horizons)

        # Performance tracking per horizon
        self.horizon_performance = {h: deque(maxlen=50) for h in horizons}

    def train_all_horizons(self, features_df, train_func):
        """
        Train models for all horizons.

        Args:
            features_df: DataFrame with features
            train_func: Function that trains model given (model, horizon, features_df)
        """
        for model, horizon in zip(self.models, self.horizons):
            print(f"Training model for horizon {horizon}...")
            train_func(model, horizon, features_df)

    def predict_proba(self, x: torch.Tensor, current_horizon: int = None) -> np.ndarray:
        """
        Predict using ensemble.

        If current_horizon is specified, use horizon-specific weighting.
        """
        predictions = []

        for model in self.models:
            with torch.no_grad():
                if hasattr(model, "predict_proba"):
                    if isinstance(x, torch.Tensor):
                        pred = model.predict_proba(x.cpu().numpy())
                    else:
                        pred = model.predict_proba(x)
                else:
                    logits = model(x)
                    pred = torch.softmax(logits, dim=-1).cpu().numpy()

            predictions.append(pred)

        predictions = np.array(predictions)

        # Adaptive weighting based on horizon
        if current_horizon is not None:
            # Weight models based on proximity to current horizon
            horizon_weights = np.array(
                [1.0 / (1.0 + abs(h - current_horizon)) for h in self.horizons]
            )
            horizon_weights /= np.sum(horizon_weights)
            weights = self.weights * horizon_weights
            weights /= np.sum(weights)
        else:
            weights = self.weights

        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return ensemble_pred

    def predict(self, x: torch.Tensor, current_horizon: int = None) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(x, current_horizon)
        return np.argmax(probs, axis=-1)

    def update_weights_by_horizon(self, horizon: int, accuracy: float):
        """Update weights based on horizon-specific performance."""
        idx = self.horizons.index(horizon)
        self.horizon_performance[horizon].append(accuracy)

        # Recalculate weights based on recent performance
        avg_performance = np.array(
            [
                (
                    np.mean(self.horizon_performance[h])
                    if len(self.horizon_performance[h]) > 0
                    else 0.5
                )
                for h in self.horizons
            ]
        )

        # Softmax with temperature
        exp_perf = np.exp(avg_performance / 0.3)
        self.weights = exp_perf / np.sum(exp_perf)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("ENSEMBLE META-LEARNER DEMONSTRATION")
    print("=" * 80)

    # Simulate models with different behaviors
    from bayesian_online import DirichletMultinomialClassifier

    # Create base models
    print("\n1. Creating base models...")
    model1 = DirichletMultinomialClassifier(num_classes=3, alpha_prior=1.0)
    model2 = DirichletMultinomialClassifier(num_classes=3, alpha_prior=2.0)
    model3 = DirichletMultinomialClassifier(num_classes=3, alpha_prior=0.5)

    # Train with different data (simulate different specializations)
    observations1 = np.array([2, 2, 2, 1, 2, 2, 0, 2])  # Biased to class 2
    observations2 = np.array([1, 1, 2, 1, 0, 1, 1, 2])  # Biased to class 1
    observations3 = np.array([0, 1, 2, 0, 1, 2, 0, 1])  # More balanced

    for obs in observations1:
        model1.update(obs)
    for obs in observations2:
        model2.update(obs)
    for obs in observations3:
        model3.update(obs)

    print(f"Model 1 probs: {model1.predict_proba()}")
    print(f"Model 2 probs: {model2.predict_proba()}")
    print(f"Model 3 probs: {model3.predict_proba()}")

    # Test weighted ensemble
    print("\n2. Weighted Ensemble")
    print("-" * 80)

    ensemble = WeightedEnsemble([model1, model2, model3])
    print(f"Initial weights: {ensemble.weights}")

    # Dummy input (not used by Bayesian models)
    x_dummy = torch.randn(5, 10, 16)

    ensemble_probs = ensemble.predict_proba(x_dummy)
    print(f"Ensemble probabilities: {ensemble_probs[0]}")
    print(f"Ensemble prediction: {ensemble.predict(x_dummy)[0]}")

    # Update weights based on performance
    y_true = np.array([2, 2, 1, 2, 2])
    ensemble.update_weights(x_dummy, y_true, method="accuracy")
    print(f"\nUpdated weights: {ensemble.weights}")

    # Test multi-horizon ensemble
    print("\n3. Multi-Horizon Ensemble")
    print("-" * 80)

    horizons = [10, 50, 100]

    # Create simple mock model class for demonstration
    class MockModel:
        def __init__(self, horizon):
            self.horizon = horizon
            self.base_probs = np.random.dirichlet([1, 1, 1])

        def predict_proba(self, x):
            batch_size = len(x) if hasattr(x, "__len__") else 1
            return np.tile(self.base_probs, (batch_size, 1))

    models = [MockModel(h) for h in horizons]
    multi_ensemble = MultiHorizonEnsemble.__new__(MultiHorizonEnsemble)
    multi_ensemble.horizons = horizons
    multi_ensemble.models = models
    multi_ensemble.weights = np.ones(len(horizons)) / len(horizons)
    multi_ensemble.horizon_performance = {h: deque(maxlen=50) for h in horizons}

    print(f"Horizons: {multi_ensemble.horizons}")
    print(f"Initial weights: {multi_ensemble.weights}")

    # Test prediction with horizon-specific weighting
    x_test = np.random.randn(3, 10)
    pred_short = multi_ensemble.predict_proba(x_test, current_horizon=10)
    pred_long = multi_ensemble.predict_proba(x_test, current_horizon=100)

    print(f"\nPrediction for horizon=10: {pred_short[0]}")
    print(f"Prediction for horizon=100: {pred_long[0]}")

    # Update performance
    multi_ensemble.update_weights_by_horizon(10, 0.7)
    multi_ensemble.update_weights_by_horizon(50, 0.6)
    multi_ensemble.update_weights_by_horizon(100, 0.8)

    print(f"\nUpdated weights after performance feedback: {multi_ensemble.weights}")

    print("\n" + "=" * 80)
    print("Key Advantages:")
    print("  • Combines diverse model architectures")
    print("  • Adaptive weighting based on performance")
    print("  • Horizon-specific specialization")
    print("  • Reduces overfitting through diversity")
    print("=" * 80)
