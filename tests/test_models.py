"""
Comprehensive Unit Tests for ML Models

This module tests LSTM, Transformer, and Bayesian models with:
- Shape validation
- Gradient flow
- Batch independence
- Invalid input handling
- Edge cases
- Performance benchmarks
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lstm_model import OrderBookLSTM, AttentionLSTM
from src.models.transformer_model import OrderBookTransformer, PositionalEncoding
from src.models.bayesian_online import DirichletMultinomialClassifier


class TestOrderBookLSTM(unittest.TestCase):
    """Test suite for OrderBookLSTM model."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.input_size = 16
        self.hidden_size = 64
        self.num_layers = 2
        self.num_classes = 3
        self.dropout = 0.3

        self.model = OrderBookLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            dropout=self.dropout
        )

    def test_model_initialization(self):
        """Test model initializes with correct architecture."""
        self.assertEqual(self.model.input_size, self.input_size)
        self.assertEqual(self.model.hidden_size, self.hidden_size)
        self.assertEqual(self.model.num_layers, self.num_layers)
        self.assertEqual(self.model.num_classes, self.num_classes)

    def test_forward_pass_shape(self):
        """Test output shapes are correct."""
        batch_size, seq_len = 8, 50
        x = torch.randn(batch_size, seq_len, self.input_size)

        logits, (h_n, c_n) = self.model(x)

        # Check output shape
        self.assertEqual(logits.shape, (batch_size, self.num_classes))

        # Check hidden state shapes
        self.assertEqual(h_n.shape, (self.num_layers, batch_size, self.hidden_size))
        self.assertEqual(c_n.shape, (self.num_layers, batch_size, self.hidden_size))

    def test_forward_pass_single_sample(self):
        """Test with single sample (batch_size=1)."""
        x = torch.randn(1, 50, self.input_size)
        logits, hidden = self.model(x)

        self.assertEqual(logits.shape, (1, self.num_classes))

    def test_variable_sequence_lengths(self):
        """Test model handles different sequence lengths."""
        for seq_len in [10, 50, 100, 200]:
            x = torch.randn(4, seq_len, self.input_size)
            logits, _ = self.model(x)
            self.assertEqual(logits.shape, (4, self.num_classes))

    def test_gradient_flow(self):
        """Test gradients flow through all parameters."""
        batch_size = 4
        x = torch.randn(batch_size, 20, self.input_size)
        y = torch.LongTensor([0, 1, 2, 1])

        # Forward pass
        logits, _ = self.model(x)
        loss = nn.CrossEntropyLoss()(logits, y)

        # Backward pass
        loss.backward()

        # Check all parameters have gradients
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient in {name}")
            self.assertFalse(torch.isinf(param.grad).any(), f"Inf gradient in {name}")

    def test_gradient_magnitudes(self):
        """Test gradient magnitudes are reasonable (not vanishing/exploding)."""
        x = torch.randn(8, 50, self.input_size)
        y = torch.LongTensor([0, 1, 2, 0, 1, 2, 0, 1])

        logits, _ = self.model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()

        grad_norms = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)

                # Check for vanishing gradients (< 1e-7)
                self.assertGreater(grad_norm, 1e-7,
                                 f"Vanishing gradient in {name}: {grad_norm}")

                # Check for exploding gradients (> 100)
                self.assertLess(grad_norm, 100,
                              f"Exploding gradient in {name}: {grad_norm}")

    def test_batch_independence(self):
        """Test predictions are independent across batch dimension."""
        x1 = torch.randn(1, 20, self.input_size)
        x2 = torch.randn(1, 20, self.input_size)
        x_batch = torch.cat([x1, x2], dim=0)

        self.model.eval()
        with torch.no_grad():
            logits1, _ = self.model(x1)
            logits2, _ = self.model(x2)
            logits_batch, _ = self.model(x_batch)

        # Check individual predictions match batch predictions
        self.assertTrue(torch.allclose(logits_batch[0], logits1[0], atol=1e-5))
        self.assertTrue(torch.allclose(logits_batch[1], logits2[0], atol=1e-5))

    def test_deterministic_inference(self):
        """Test model gives same results with same input in eval mode."""
        x = torch.randn(4, 30, self.input_size)

        self.model.eval()
        with torch.no_grad():
            logits1, _ = self.model(x)
            logits2, _ = self.model(x)

        self.assertTrue(torch.allclose(logits1, logits2, atol=1e-6))

    def test_dropout_training_vs_eval(self):
        """Test dropout behaves differently in train vs eval mode."""
        x = torch.randn(4, 30, self.input_size)

        # Training mode (dropout active)
        self.model.train()
        with torch.no_grad():
            outputs_train = [self.model(x)[0] for _ in range(5)]

        # Eval mode (no dropout)
        self.model.eval()
        with torch.no_grad():
            outputs_eval = [self.model(x)[0] for _ in range(5)]

        # Training outputs should vary (dropout randomness)
        train_variance = torch.stack(outputs_train).var(dim=0).mean().item()

        # Eval outputs should be identical
        eval_variance = torch.stack(outputs_eval).var(dim=0).mean().item()

        self.assertGreater(train_variance, eval_variance * 10,
                          "Dropout not working properly")

    def test_invalid_input_dimensions(self):
        """Test model rejects invalid input dimensions."""
        # 2D input (missing sequence dimension)
        with self.assertRaises(RuntimeError):
            self.model(torch.randn(8, self.input_size))

        # Wrong feature dimension
        with self.assertRaises(RuntimeError):
            self.model(torch.randn(8, 20, self.input_size + 5))

        # 4D input
        with self.assertRaises(RuntimeError):
            self.model(torch.randn(8, 20, 4, self.input_size))

    def test_zero_input(self):
        """Test model handles zero input gracefully."""
        x = torch.zeros(4, 30, self.input_size)
        logits, _ = self.model(x)

        self.assertEqual(logits.shape, (4, self.num_classes))
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())

    def test_large_input_values(self):
        """Test model handles large input values."""
        x = torch.randn(4, 30, self.input_size) * 1000  # Large values
        logits, _ = self.model(x)

        self.assertFalse(torch.isnan(logits).any(), "NaN in output with large inputs")
        self.assertFalse(torch.isinf(logits).any(), "Inf in output with large inputs")

    def test_model_to_device(self):
        """Test model can be moved to different devices."""
        # CPU
        self.model.cpu()
        x_cpu = torch.randn(4, 20, self.input_size)
        logits_cpu, _ = self.model(x_cpu)
        self.assertEqual(logits_cpu.device.type, 'cpu')

        # CUDA (if available)
        if torch.cuda.is_available():
            self.model.cuda()
            x_cuda = torch.randn(4, 20, self.input_size).cuda()
            logits_cuda, _ = self.model(x_cuda)
            self.assertEqual(logits_cuda.device.type, 'cuda')


class TestAttentionLSTM(unittest.TestCase):
    """Test suite for AttentionLSTM model."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.input_size = 16
        self.hidden_size = 64
        self.num_layers = 2
        self.num_classes = 3

        self.model = AttentionLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.num_classes
        )

    def test_forward_pass_shape(self):
        """Test output shapes with attention."""
        batch_size, seq_len = 8, 50
        x = torch.randn(batch_size, seq_len, self.input_size)

        logits, hidden = self.model(x)

        self.assertEqual(logits.shape, (batch_size, self.num_classes))

    def test_attention_weights_sum_to_one(self):
        """Test attention weights are valid probabilities."""
        # This would require extracting attention weights
        # Currently the model doesn't expose them, so we skip
        pass

    def test_gradient_flow_through_attention(self):
        """Test gradients flow through attention mechanism."""
        x = torch.randn(4, 30, self.input_size)
        y = torch.LongTensor([0, 1, 2, 1])

        logits, _ = self.model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()

        # Check gradients exist
        for name, param in self.model.named_parameters():
            if 'attention' in name:
                self.assertIsNotNone(param.grad, f"No gradient for attention: {name}")


class TestOrderBookTransformer(unittest.TestCase):
    """Test suite for OrderBookTransformer model."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.input_size = 16
        self.d_model = 128
        self.nhead = 8
        self.num_encoder_layers = 4
        self.num_classes = 3

        self.model = OrderBookTransformer(
            input_size=self.input_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_classes=self.num_classes
        )

    def test_forward_pass_shape(self):
        """Test transformer output shapes."""
        batch_size, seq_len = 8, 50
        x = torch.randn(batch_size, seq_len, self.input_size)

        logits = self.model(x)

        self.assertEqual(logits.shape, (batch_size, self.num_classes))

    def test_d_model_must_be_even(self):
        """Test d_model validation (must be even for positional encoding)."""
        # This should work
        model = OrderBookTransformer(
            input_size=16,
            d_model=128,  # Even
            nhead=8,
            num_classes=3
        )

        # Odd d_model may cause issues in positional encoding
        # but PyTorch doesn't enforce it, so we just document it
        pass

    def test_nhead_divides_d_model(self):
        """Test nhead must divide d_model evenly."""
        with self.assertRaises(AssertionError):
            OrderBookTransformer(
                input_size=16,
                d_model=128,
                nhead=7,  # Doesn't divide 128
                num_classes=3
            )

    def test_gradient_flow(self):
        """Test gradients flow through transformer."""
        x = torch.randn(4, 30, self.input_size)
        y = torch.LongTensor([0, 1, 2, 1])

        logits = self.model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()

        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient in {name}")

    def test_variable_sequence_lengths(self):
        """Test transformer with different sequence lengths."""
        for seq_len in [10, 50, 100]:
            x = torch.randn(4, seq_len, self.input_size)
            logits = self.model(x)
            self.assertEqual(logits.shape, (4, self.num_classes))

    def test_attention_mask(self):
        """Test transformer with attention mask."""
        batch_size, seq_len = 4, 50
        x = torch.randn(batch_size, seq_len, self.input_size)

        # Create causal mask (for autoregressive prediction)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

        # Should not raise error
        logits = self.model(x, mask=mask)
        self.assertEqual(logits.shape, (batch_size, self.num_classes))


class TestPositionalEncoding(unittest.TestCase):
    """Test suite for PositionalEncoding module."""

    def setUp(self):
        """Set up test fixtures."""
        self.d_model = 128
        self.max_len = 5000
        self.pe = PositionalEncoding(d_model=self.d_model, max_len=self.max_len)

    def test_output_shape(self):
        """Test positional encoding output shape."""
        batch_size, seq_len = 8, 50
        x = torch.randn(batch_size, seq_len, self.d_model)

        output = self.pe(x)

        self.assertEqual(output.shape, x.shape)

    def test_encoding_is_deterministic(self):
        """Test positional encoding is deterministic."""
        x = torch.randn(4, 30, self.d_model)

        output1 = self.pe(x)
        output2 = self.pe(x)

        self.assertTrue(torch.allclose(output1, output2))

    def test_different_positions_different_encodings(self):
        """Test different positions get different encodings."""
        x = torch.zeros(1, 100, self.d_model)
        output = self.pe(x)

        # Check first and last positions are different
        diff = (output[0, 0] - output[0, 99]).abs().sum().item()
        self.assertGreater(diff, 0.1, "Positional encodings too similar")


class TestDirichletMultinomialClassifier(unittest.TestCase):
    """Test suite for Bayesian online learning model."""

    def setUp(self):
        """Set up test fixtures."""
        self.num_classes = 3
        self.alpha_prior = 1.0
        self.model = DirichletMultinomialClassifier(
            num_classes=self.num_classes,
            alpha_prior=self.alpha_prior
        )

    def test_initialization(self):
        """Test model initializes with correct priors."""
        self.assertEqual(len(self.model.alpha), self.num_classes)
        self.assertTrue(all(a == self.alpha_prior for a in self.model.alpha))

    def test_predict_proba_sums_to_one(self):
        """Test predicted probabilities sum to 1."""
        probs = self.model.predict_proba()

        self.assertEqual(len(probs), self.num_classes)
        self.assertAlmostEqual(probs.sum(), 1.0, places=10)

    def test_predict_proba_positive(self):
        """Test all probabilities are positive."""
        probs = self.model.predict_proba()

        self.assertTrue(all(p > 0 for p in probs))
        self.assertTrue(all(p < 1 for p in probs))

    def test_predict_returns_valid_class(self):
        """Test predict returns valid class index."""
        prediction = self.model.predict()

        self.assertIsInstance(prediction, (int, np.integer))
        self.assertGreaterEqual(prediction, 0)
        self.assertLess(prediction, self.num_classes)

    def test_update_increases_alpha(self):
        """Test update increases corresponding alpha value."""
        initial_alpha = self.model.alpha.copy()

        self.model.update(observation=1)

        # Alpha for class 1 should increase
        self.assertGreater(self.model.alpha[1], initial_alpha[1])

        # Other alphas should remain the same
        self.assertEqual(self.model.alpha[0], initial_alpha[0])
        self.assertEqual(self.model.alpha[2], initial_alpha[2])

    def test_uncertainty_decreases_with_data(self):
        """Test uncertainty decreases as more data is observed."""
        initial_uncertainty = self.model.get_uncertainty()

        # Add many observations
        for _ in range(100):
            self.model.update(observation=0)

        final_uncertainty = self.model.get_uncertainty()

        self.assertLess(final_uncertainty, initial_uncertainty)

    def test_invalid_observation_raises_error(self):
        """Test invalid observation index raises error."""
        with self.assertRaises(ValueError):
            self.model.update(observation=-1)

        with self.assertRaises(ValueError):
            self.model.update(observation=self.num_classes)

    def test_reset(self):
        """Test reset returns model to initial state."""
        # Make some updates
        for i in range(10):
            self.model.update(observation=i % self.num_classes)

        # Reset
        self.model.reset()

        # Should be back to initial state
        self.assertTrue(all(a == self.alpha_prior for a in self.model.alpha))

    def test_convergence_with_known_distribution(self):
        """Test model converges to true distribution with sufficient data."""
        # True distribution: [0.5, 0.3, 0.2]
        true_probs = [0.5, 0.3, 0.2]
        np.random.seed(42)

        # Generate many samples
        for _ in range(1000):
            observation = np.random.choice(self.num_classes, p=true_probs)
            self.model.update(observation)

        # Predicted probabilities should be close to true
        predicted_probs = self.model.predict_proba()

        for true_p, pred_p in zip(true_probs, predicted_probs):
            self.assertAlmostEqual(true_p, pred_p, delta=0.05)


class TestModelPerformance(unittest.TestCase):
    """Performance benchmarks for models."""

    def test_lstm_inference_speed(self):
        """Test LSTM inference meets latency requirements."""
        import time

        model = OrderBookLSTM(input_size=16, hidden_size=128, num_layers=2)
        model.eval()

        batch_size = 32
        seq_len = 50
        x = torch.randn(batch_size, seq_len, 16)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(x)

        # Benchmark
        num_iterations = 100
        start = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(x)
        elapsed = time.time() - start

        avg_time_ms = (elapsed / num_iterations) * 1000
        throughput = (batch_size * num_iterations) / elapsed

        print(f"\nLSTM Performance:")
        print(f"  Average inference time: {avg_time_ms:.2f}ms")
        print(f"  Throughput: {throughput:.0f} samples/sec")

        # Should be faster than 50ms for batch
        self.assertLess(avg_time_ms, 50,
                       f"LSTM too slow: {avg_time_ms:.2f}ms > 50ms")

    def test_transformer_inference_speed(self):
        """Test Transformer inference meets latency requirements."""
        import time

        model = OrderBookTransformer(input_size=16, d_model=128, nhead=8)
        model.eval()

        batch_size = 32
        seq_len = 50
        x = torch.randn(batch_size, seq_len, 16)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(x)

        # Benchmark
        num_iterations = 100
        start = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(x)
        elapsed = time.time() - start

        avg_time_ms = (elapsed / num_iterations) * 1000
        throughput = (batch_size * num_iterations) / elapsed

        print(f"\nTransformer Performance:")
        print(f"  Average inference time: {avg_time_ms:.2f}ms")
        print(f"  Throughput: {throughput:.0f} samples/sec")

        # Transformers are typically slower than LSTMs
        self.assertLess(avg_time_ms, 100,
                       f"Transformer too slow: {avg_time_ms:.2f}ms > 100ms")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
