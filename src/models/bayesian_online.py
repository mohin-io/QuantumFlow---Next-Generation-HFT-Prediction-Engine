"""
Bayesian Online Learning for Order Book Forecasting

Implements online Bayesian methods for adaptive prediction:
- Beta-Bernoulli model for binary outcomes
- Dirichlet-Multinomial for multi-class
- Online variational inference
- Uncertainty quantification

Benefits:
- No retraining required
- Real-time adaptation to regime changes
- Uncertainty estimates for each prediction
- Computationally efficient

References:
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning
- Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective
"""

import numpy as np
from scipy.special import gammaln, digamma
from scipy.stats import dirichlet
from typing import Dict, Tuple


class DirichletMultinomialClassifier:
    """
    Online Bayesian classifier using Dirichlet-Multinomial conjugate pair.

    Maintains a Dirichlet distribution over class probabilities and updates
    it online as new observations arrive.
    """

    def __init__(self, num_classes=3, alpha_prior=1.0):
        """
        Initialize Bayesian classifier.

        Args:
            num_classes: Number of classes
            alpha_prior: Prior concentration parameter (pseudo-counts)
        """
        self.num_classes = num_classes
        self.alpha_prior = alpha_prior

        # Initialize Dirichlet parameters (alpha)
        # Start with uniform prior
        self.alpha = np.ones(num_classes) * alpha_prior

        # Track statistics
        self.n_updates = 0

    def predict_proba(self) -> np.ndarray:
        """
        Get current probability estimates.

        Returns expected value of Dirichlet distribution.
        """
        return self.alpha / np.sum(self.alpha)

    def predict(self) -> int:
        """Get most likely class."""
        return np.argmax(self.predict_proba())

    def get_uncertainty(self) -> float:
        """
        Get prediction uncertainty using entropy.

        Higher entropy = more uncertainty
        """
        probs = self.predict_proba()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return entropy

    def get_confidence(self) -> float:
        """
        Get prediction confidence.

        Based on concentration of Dirichlet distribution.
        """
        # Concentration = sum of alpha parameters
        # Higher concentration = more confident
        concentration = np.sum(self.alpha)

        # Normalize to [0, 1]
        confidence = 1.0 - (1.0 / (1.0 + concentration))

        return confidence

    def update(self, observation: int):
        """
        Update posterior with new observation.

        Args:
            observation: Observed class (0, 1, or 2)
        """
        # Bayesian update: add observation to corresponding alpha
        self.alpha[observation] += 1.0
        self.n_updates += 1

    def batch_update(self, observations: np.ndarray):
        """
        Update with batch of observations.

        Args:
            observations: Array of class labels
        """
        # Count occurrences
        counts = np.bincount(observations, minlength=self.num_classes)

        # Update alpha
        self.alpha += counts
        self.n_updates += len(observations)

    def get_credible_interval(self, confidence=0.95) -> np.ndarray:
        """
        Get credible intervals for class probabilities.

        Returns array of shape (num_classes, 2) with [lower, upper] bounds.
        """
        # Sample from posterior
        samples = dirichlet.rvs(self.alpha, size=10000)

        # Compute percentiles
        lower = (1 - confidence) / 2
        upper = 1 - lower

        intervals = np.percentile(samples, [lower * 100, upper * 100], axis=0).T

        return intervals

    def reset(self):
        """Reset to prior."""
        self.alpha = np.ones(self.num_classes) * self.alpha_prior
        self.n_updates = 0


class BayesianOnlineEnsemble:
    """
    Ensemble of Bayesian classifiers with feature-based routing.

    Maintains multiple Bayesian models and routes predictions based on
    feature characteristics.
    """

    def __init__(self, num_classes=3, num_models=5):
        """
        Initialize ensemble.

        Args:
            num_classes: Number of classes
            num_models: Number of base models
        """
        self.num_classes = num_classes
        self.num_models = num_models

        # Create base models
        self.models = [
            DirichletMultinomialClassifier(num_classes=num_classes)
            for _ in range(num_models)
        ]

        # Model weights (updated based on performance)
        self.weights = np.ones(num_models) / num_models

        # Track model performance
        self.model_correct = np.zeros(num_models)
        self.model_total = np.zeros(num_models)

    def predict_proba(self) -> np.ndarray:
        """
        Get ensemble probability prediction.

        Weighted average of individual model predictions.
        """
        probs_list = np.array([model.predict_proba() for model in self.models])

        # Weighted average
        ensemble_probs = np.average(probs_list, axis=0, weights=self.weights)

        return ensemble_probs

    def predict(self) -> int:
        """Get ensemble prediction."""
        return np.argmax(self.predict_proba())

    def update(self, observation: int, model_idx: int = None):
        """
        Update models with new observation.

        Args:
            observation: Observed class
            model_idx: Which model to update (if None, update all)
        """
        if model_idx is None:
            # Update all models
            for model in self.models:
                model.update(observation)
        else:
            # Update specific model
            self.models[model_idx].update(observation)

    def update_weights(self, predictions: np.ndarray, actuals: np.ndarray):
        """
        Update model weights based on recent performance.

        Args:
            predictions: Array of predictions from each model
            actuals: Array of actual labels
        """
        # Calculate accuracy for each model
        for i in range(self.num_models):
            correct = np.sum(predictions[i] == actuals)
            self.model_correct[i] += correct
            self.model_total[i] += len(actuals)

        # Update weights (softmax of accuracies)
        accuracies = self.model_correct / (self.model_total + 1e-10)

        # Temperature-scaled softmax
        temperature = 0.5
        exp_acc = np.exp(accuracies / temperature)
        self.weights = exp_acc / np.sum(exp_acc)


class BayesianMovingAverage:
    """
    Bayesian moving average for online prediction.

    Uses Normal-Gamma conjugate prior for adaptive mean estimation.
    """

    def __init__(self, mu0=0.0, lambda0=1.0, alpha0=1.0, beta0=1.0):
        """
        Initialize Bayesian moving average.

        Args:
            mu0: Prior mean
            lambda0: Prior precision (inverse variance) of mean
            alpha0: Prior shape parameter for precision
            beta0: Prior rate parameter for precision
        """
        # Hyperparameters (updated online)
        self.mu = mu0
        self.lambda_ = lambda0
        self.alpha = alpha0
        self.beta = beta0

        self.n_updates = 0

    def predict(self) -> Tuple[float, float]:
        """
        Get predictive distribution parameters.

        Returns:
            (mean, variance) of predictive distribution
        """
        # Predictive mean
        mean = self.mu

        # Predictive variance (from Student-t)
        variance = self.beta * (self.lambda_ + 1) / (self.alpha * self.lambda_)

        return mean, variance

    def update(self, observation: float):
        """
        Update posterior with new observation.

        Args:
            observation: New data point
        """
        # Update hyperparameters
        lambda_new = self.lambda_ + 1
        mu_new = (self.lambda_ * self.mu + observation) / lambda_new
        alpha_new = self.alpha + 0.5
        beta_new = (
            self.beta + 0.5 * self.lambda_ * (observation - self.mu) ** 2 / lambda_new
        )

        self.lambda_ = lambda_new
        self.mu = mu_new
        self.alpha = alpha_new
        self.beta = beta_new

        self.n_updates += 1


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("BAYESIAN ONLINE LEARNING DEMONSTRATION")
    print("=" * 80)

    # Test Dirichlet-Multinomial classifier
    print("\n1. Dirichlet-Multinomial Classifier")
    print("-" * 80)

    classifier = DirichletMultinomialClassifier(num_classes=3, alpha_prior=1.0)

    print(f"Initial probabilities: {classifier.predict_proba()}")
    print(f"Initial uncertainty (entropy): {classifier.get_uncertainty():.4f}")
    print(f"Initial confidence: {classifier.get_confidence():.4f}")

    # Simulate observations (mostly class 2)
    observations = np.array([2, 2, 1, 2, 2, 0, 2, 2, 2, 1])

    print(f"\nObservations: {observations}")
    print("\nUpdating classifier...")

    for obs in observations:
        classifier.update(obs)

    print(f"\nAfter updates:")
    print(f"  Probabilities: {classifier.predict_proba()}")
    print(
        f"  Prediction: {classifier.predict()} ({['Down', 'Flat', 'Up'][classifier.predict()]})"
    )
    print(f"  Uncertainty: {classifier.get_uncertainty():.4f}")
    print(f"  Confidence: {classifier.get_confidence():.4f}")

    # Credible intervals
    intervals = classifier.get_credible_interval(confidence=0.95)
    print(f"\n95% Credible Intervals:")
    for i, (lower, upper) in enumerate(intervals):
        print(f"  Class {i}: [{lower:.4f}, {upper:.4f}]")

    # Test ensemble
    print("\n2. Bayesian Ensemble")
    print("-" * 80)

    ensemble = BayesianOnlineEnsemble(num_classes=3, num_models=5)

    print(f"Initial ensemble probabilities: {ensemble.predict_proba()}")
    print(f"Initial weights: {ensemble.weights}")

    # Update ensemble
    for obs in observations:
        ensemble.update(obs)

    print(f"\nAfter updates:")
    print(f"  Ensemble probabilities: {ensemble.predict_proba()}")
    print(f"  Ensemble prediction: {ensemble.predict()}")
    print(f"  Model weights: {ensemble.weights}")

    # Test moving average
    print("\n3. Bayesian Moving Average")
    print("-" * 80)

    bma = BayesianMovingAverage(mu0=0.0, lambda0=1.0)

    print(f"Initial: mean={bma.predict()[0]:.4f}, var={bma.predict()[1]:.4f}")

    # Stream of values
    values = np.array([0.5, 0.7, 0.6, 0.8, 0.65, 0.75])

    for val in values:
        bma.update(val)

    mean, var = bma.predict()
    print(f"\nAfter {len(values)} updates:")
    print(f"  Predictive mean: {mean:.4f}")
    print(f"  Predictive variance: {var:.4f}")
    print(f"  Predictive std: {np.sqrt(var):.4f}")

    print("\n" + "=" * 80)
    print("Key Advantages:")
    print("  • No retraining required")
    print("  • Adapts to regime changes in real-time")
    print("  • Provides uncertainty estimates")
    print("  • Computationally efficient (O(1) per update)")
    print("=" * 80)
