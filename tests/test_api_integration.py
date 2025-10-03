"""
Comprehensive API Integration Tests

Tests the FastAPI prediction service with real HTTP requests using TestClient.
Covers all endpoints, error cases, rate limiting, validation, and performance.

Run with: pytest tests/test_api_integration.py -v
"""

import asyncio
import json
import time
import unittest
from typing import List

import pytest
from fastapi.testclient import TestClient

# Import the app from the improved API
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.prediction_service_improved import app, metrics_collector


class TestAPIHealthEndpoints(unittest.TestCase):
    """Test health check and status endpoints."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_root_endpoint(self):
        """Test root endpoint returns API information."""
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("service", data)
        self.assertIn("version", data)
        self.assertIn("endpoints", data)
        self.assertEqual(data["status"], "operational")

    def test_health_check_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Validate response structure
        self.assertEqual(data["status"], "healthy")
        self.assertIn("timestamp", data)
        self.assertIn("version", data)
        self.assertIn("models_loaded", data)
        self.assertIn("uptime_seconds", data)

        # Check data types
        self.assertIsInstance(data["models_loaded"], list)
        self.assertIsInstance(data["uptime_seconds"], (int, float))
        self.assertGreaterEqual(data["uptime_seconds"], 0)

    def test_models_list_endpoint(self):
        """Test models listing endpoint."""
        response = self.client.get("/models")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("models", data)
        self.assertIsInstance(data["models"], list)

        # Check at least dummy model is loaded
        model_names = [m["name"] for m in data["models"]]
        self.assertIn("dummy", model_names)

        # Validate model structure
        for model in data["models"]:
            self.assertIn("name", model)
            self.assertIn("type", model)
            self.assertIn("loaded", model)
            self.assertTrue(model["loaded"])


class TestAPIPredictionEndpoint(unittest.TestCase):
    """Test prediction endpoint with various scenarios."""

    def setUp(self):
        """Set up test client and sample data."""
        self.client = TestClient(app)

        # Valid sample snapshot
        self.valid_snapshot = {
            "timestamp": 1000,
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "bids": [[50000.0, 1.5], [49999.0, 2.0], [49998.0, 1.8]],
            "asks": [[50001.0, 1.2], [50002.0, 1.8], [50003.0, 2.1]],
            "sequence": 12345,
        }

        # Valid prediction request
        self.valid_request = {
            "snapshots": [self.valid_snapshot for _ in range(10)],
            "model_name": "dummy",
            "include_features": False,
        }

    def test_valid_prediction_request(self):
        """Test successful prediction with valid data."""
        response = self.client.post("/predict", json=self.valid_request)

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Validate response structure
        self.assertIn("prediction", data)
        self.assertIn("probabilities", data)
        self.assertIn("confidence", data)
        self.assertIn("model_name", data)
        self.assertIn("latency_ms", data)
        self.assertIn("cache_hit", data)
        self.assertIn("timestamp", data)

        # Validate prediction value
        self.assertIn(data["prediction"], ["up", "down", "flat"])

        # Validate probabilities
        self.assertIsInstance(data["probabilities"], dict)
        self.assertIn("up", data["probabilities"])
        self.assertIn("down", data["probabilities"])
        self.assertIn("flat", data["probabilities"])

        # Probabilities should sum to ~1.0
        prob_sum = sum(data["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)

        # Validate confidence
        self.assertGreaterEqual(data["confidence"], 0.0)
        self.assertLessEqual(data["confidence"], 1.0)

        # Validate latency
        self.assertGreater(data["latency_ms"], 0)

        # Model name should match request
        self.assertEqual(data["model_name"], "dummy")

    def test_cache_functionality(self):
        """Test that cache works correctly."""
        # First request (cache miss)
        response1 = self.client.post("/predict", json=self.valid_request)
        data1 = response1.json()
        self.assertFalse(data1["cache_hit"])

        # Second identical request (should hit cache)
        response2 = self.client.post("/predict", json=self.valid_request)
        data2 = response2.json()
        self.assertTrue(data2["cache_hit"])

        # Cache hit should be much faster
        self.assertLess(data2["latency_ms"], data1["latency_ms"])

    def test_invalid_model_name(self):
        """Test prediction with non-existent model."""
        invalid_request = self.valid_request.copy()
        invalid_request["model_name"] = "nonexistent_model_xyz"

        response = self.client.post("/predict", json=invalid_request)

        self.assertEqual(response.status_code, 404)
        data = response.json()

        self.assertIn("error", data["detail"])
        self.assertEqual(data["detail"]["error"], "model_not_found")
        self.assertIn("available_models", data["detail"])
        self.assertIn("hint", data["detail"])

    def test_empty_snapshots_rejected(self):
        """Test that empty snapshots list is rejected."""
        invalid_request = self.valid_request.copy()
        invalid_request["snapshots"] = []

        response = self.client.post("/predict", json=invalid_request)

        self.assertEqual(response.status_code, 422)  # Validation error

    def test_too_many_snapshots_rejected(self):
        """Test that excessive snapshots are rejected."""
        invalid_request = self.valid_request.copy()
        invalid_request["snapshots"] = [self.valid_snapshot] * 1001  # Max is 1000

        response = self.client.post("/predict", json=invalid_request)

        self.assertEqual(response.status_code, 422)  # Validation error

    def test_invalid_snapshot_structure(self):
        """Test that malformed snapshots are rejected."""
        # Missing required fields
        invalid_snapshot = {
            "timestamp": 1000,
            "exchange": "binance",
            # Missing symbol, bids, asks
        }

        invalid_request = {"snapshots": [invalid_snapshot], "model_name": "dummy"}

        response = self.client.post("/predict", json=invalid_request)

        self.assertEqual(response.status_code, 422)  # Validation error

    def test_negative_prices_rejected(self):
        """Test that negative prices are rejected."""
        invalid_snapshot = self.valid_snapshot.copy()
        invalid_snapshot["bids"] = [[-100.0, 1.5]]  # Negative price

        invalid_request = {"snapshots": [invalid_snapshot], "model_name": "dummy"}

        response = self.client.post("/predict", json=invalid_request)

        self.assertEqual(response.status_code, 422)

    def test_crossed_book_rejected(self):
        """Test that crossed order book (bid > ask) is rejected."""
        invalid_snapshot = self.valid_snapshot.copy()
        invalid_snapshot["bids"] = [[50010.0, 1.5]]  # Bid higher than ask
        invalid_snapshot["asks"] = [[50000.0, 1.2]]

        invalid_request = {"snapshots": [invalid_snapshot], "model_name": "dummy"}

        response = self.client.post("/predict", json=invalid_request)

        # Should be rejected by validator
        self.assertEqual(response.status_code, 422)

    def test_invalid_exchange_name_rejected(self):
        """Test that invalid exchange names are rejected."""
        invalid_snapshot = self.valid_snapshot.copy()
        invalid_snapshot["exchange"] = "invalid@exchange!"  # Special chars

        invalid_request = {"snapshots": [invalid_snapshot], "model_name": "dummy"}

        response = self.client.post("/predict", json=invalid_request)

        self.assertEqual(response.status_code, 422)

    def test_invalid_symbol_rejected(self):
        """Test that invalid symbols are rejected."""
        invalid_snapshot = self.valid_snapshot.copy()
        invalid_snapshot["symbol"] = "invalid symbol!"  # Lowercase + space

        invalid_request = {"snapshots": [invalid_snapshot], "model_name": "dummy"}

        response = self.client.post("/predict", json=invalid_request)

        self.assertEqual(response.status_code, 422)

    def test_prediction_with_features(self):
        """Test prediction request with include_features=True."""
        request_with_features = self.valid_request.copy()
        request_with_features["include_features"] = True

        response = self.client.post("/predict", json=request_with_features)

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Features should be None for now (not implemented yet)
        # When implemented, change to assertIsNotNone
        self.assertIsNone(data.get("features"))


class TestAPIRateLimiting(unittest.TestCase):
    """Test rate limiting functionality."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

        self.valid_request = {
            "snapshots": [
                {
                    "timestamp": 1000,
                    "exchange": "binance",
                    "symbol": "BTCUSDT",
                    "bids": [[50000.0, 1.5]],
                    "asks": [[50001.0, 1.2]],
                }
            ],
            "model_name": "dummy",
        }

    def test_rate_limit_enforcement(self):
        """Test that rate limiting is enforced (10 requests/minute)."""
        # Make 11 requests rapidly (limit is 10/min)
        responses = []
        for _ in range(11):
            response = self.client.post("/predict", json=self.valid_request)
            responses.append(response)
            time.sleep(0.1)  # Small delay to avoid overwhelming

        # First 10 should succeed
        for i in range(10):
            self.assertEqual(
                responses[i].status_code, 200, f"Request {i+1} should succeed"
            )

        # 11th should be rate limited
        self.assertEqual(
            responses[10].status_code, 429, "Request 11 should be rate limited"
        )


class TestAPIMetricsEndpoint(unittest.TestCase):
    """Test metrics collection and endpoint."""

    def setUp(self):
        """Set up test client and reset metrics."""
        self.client = TestClient(app)
        metrics_collector.reset()

        self.valid_request = {
            "snapshots": [
                {
                    "timestamp": 1000,
                    "exchange": "binance",
                    "symbol": "BTCUSDT",
                    "bids": [[50000.0, 1.5]],
                    "asks": [[50001.0, 1.2]],
                }
            ],
            "model_name": "dummy",
        }

    def test_metrics_endpoint_structure(self):
        """Test metrics endpoint returns proper structure."""
        response = self.client.get("/metrics")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Validate structure
        self.assertIn("uptime_seconds", data)
        self.assertIn("total_predictions", data)
        self.assertIn("predictions_per_second", data)
        self.assertIn("latency_ms", data)
        self.assertIn("cache", data)
        self.assertIn("predictions_by_model", data)
        self.assertIn("errors", data)

        # Validate latency metrics
        latency = data["latency_ms"]
        self.assertIn("avg", latency)
        self.assertIn("p50", latency)
        self.assertIn("p95", latency)
        self.assertIn("p99", latency)

        # Validate cache metrics
        cache = data["cache"]
        self.assertIn("hits", cache)
        self.assertIn("misses", cache)
        self.assertIn("hit_rate_percent", cache)

    def test_metrics_updated_after_predictions(self):
        """Test that metrics are updated after making predictions."""
        # Get initial metrics
        initial_response = self.client.get("/metrics")
        initial_data = initial_response.json()
        initial_count = initial_data["total_predictions"]

        # Make a prediction
        self.client.post("/predict", json=self.valid_request)

        # Get updated metrics
        updated_response = self.client.get("/metrics")
        updated_data = updated_response.json()

        # Count should increase
        self.assertEqual(updated_data["total_predictions"], initial_count + 1)

        # Predictions by model should update
        self.assertIn("dummy", updated_data["predictions_by_model"])
        self.assertGreater(updated_data["predictions_by_model"]["dummy"], 0)

    def test_cache_metrics_tracking(self):
        """Test that cache metrics are tracked correctly."""
        # Reset metrics
        self.client.post("/metrics/reset")

        # First request (cache miss)
        self.client.post("/predict", json=self.valid_request)

        # Second identical request (cache hit)
        self.client.post("/predict", json=self.valid_request)

        # Check metrics
        response = self.client.get("/metrics")
        data = response.json()

        self.assertEqual(data["cache"]["hits"], 1)
        self.assertEqual(data["cache"]["misses"], 1)
        self.assertAlmostEqual(data["cache"]["hit_rate_percent"], 50.0, places=1)

    def test_metrics_reset_endpoint(self):
        """Test metrics reset functionality."""
        # Make some predictions
        for _ in range(5):
            self.client.post("/predict", json=self.valid_request)

        # Verify metrics exist
        response1 = self.client.get("/metrics")
        self.assertGreater(response1.json()["total_predictions"], 0)

        # Reset metrics
        reset_response = self.client.post("/metrics/reset")
        self.assertEqual(reset_response.status_code, 200)

        # Verify metrics are reset
        response2 = self.client.get("/metrics")
        self.assertEqual(response2.json()["total_predictions"], 0)


class TestAPIPerformance(unittest.TestCase):
    """Test API performance and latency."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

        self.valid_request = {
            "snapshots": [
                {
                    "timestamp": 1000,
                    "exchange": "binance",
                    "symbol": "BTCUSDT",
                    "bids": [[50000.0, 1.5]],
                    "asks": [[50001.0, 1.2]],
                }
            ],
            "model_name": "dummy",
        }

    def test_prediction_latency_under_threshold(self):
        """Test that predictions complete within acceptable time."""
        response = self.client.post("/predict", json=self.valid_request)

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Should be under 100ms for small request
        self.assertLess(
            data["latency_ms"],
            100,
            f"Latency {data['latency_ms']}ms exceeds 100ms threshold",
        )

    def test_throughput_capacity(self):
        """Test API can handle multiple rapid requests."""
        num_requests = 50
        start_time = time.time()

        responses = []
        for _ in range(num_requests):
            response = self.client.post("/predict", json=self.valid_request)
            responses.append(response)

        elapsed = time.time() - start_time

        # All should succeed
        success_count = sum(1 for r in responses if r.status_code == 200)
        self.assertGreaterEqual(
            success_count,
            40,  # Allow some for rate limiting
            f"Only {success_count}/{num_requests} succeeded",
        )

        # Calculate throughput
        throughput = num_requests / elapsed
        print(f"\nThroughput: {throughput:.1f} requests/second")

        # Should handle at least 10 requests/sec
        self.assertGreater(throughput, 10)

    def test_large_request_performance(self):
        """Test performance with maximum allowed snapshots."""
        large_request = self.valid_request.copy()
        large_request["snapshots"] = [self.valid_request["snapshots"][0]] * 1000

        start_time = time.time()
        response = self.client.post("/predict", json=large_request)
        elapsed = (time.time() - start_time) * 1000  # Convert to ms

        self.assertEqual(response.status_code, 200)

        # Should complete within 5 seconds (timeout threshold)
        self.assertLess(
            elapsed, 5000, f"Large request took {elapsed}ms (> 5000ms threshold)"
        )


class TestAPIErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_invalid_json_rejected(self):
        """Test that invalid JSON is rejected."""
        response = self.client.post(
            "/predict",
            data="invalid json{{{",
            headers={"Content-Type": "application/json"},
        )

        self.assertEqual(response.status_code, 422)

    def test_missing_required_fields(self):
        """Test that missing required fields are caught."""
        incomplete_request = {
            "model_name": "dummy"
            # Missing snapshots field
        }

        response = self.client.post("/predict", json=incomplete_request)

        self.assertEqual(response.status_code, 422)

    def test_wrong_data_types(self):
        """Test that wrong data types are rejected."""
        wrong_types = {
            "snapshots": "not a list",  # Should be list
            "model_name": 123,  # Should be string
        }

        response = self.client.post("/predict", json=wrong_types)

        self.assertEqual(response.status_code, 422)

    def test_get_on_post_endpoint(self):
        """Test that GET on POST endpoint returns error."""
        response = self.client.get("/predict")

        self.assertEqual(response.status_code, 405)  # Method not allowed


class TestAPIDocumentation(unittest.TestCase):
    """Test API documentation endpoints."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_openapi_docs_available(self):
        """Test that OpenAPI docs are accessible."""
        response = self.client.get("/docs")

        self.assertEqual(response.status_code, 200)

    def test_redoc_available(self):
        """Test that ReDoc is accessible."""
        response = self.client.get("/redoc")

        self.assertEqual(response.status_code, 200)

    def test_openapi_json_schema(self):
        """Test that OpenAPI JSON schema is available."""
        response = self.client.get("/openapi.json")

        self.assertEqual(response.status_code, 200)

        # Validate it's valid JSON
        data = response.json()
        self.assertIn("openapi", data)
        self.assertIn("info", data)
        self.assertIn("paths", data)


# Performance benchmark test
def test_api_benchmark():
    """Benchmark API performance (not a unit test, for manual runs)."""
    client = TestClient(app)

    request = {
        "snapshots": [
            {
                "timestamp": 1000,
                "exchange": "binance",
                "symbol": "BTCUSDT",
                "bids": [[50000.0, 1.5]],
                "asks": [[50001.0, 1.2]],
            }
        ],
        "model_name": "dummy",
    }

    print("\n" + "=" * 80)
    print("API Performance Benchmark")
    print("=" * 80)

    # Warmup
    for _ in range(10):
        client.post("/predict", json=request)

    # Benchmark
    num_requests = 100
    latencies = []

    for _ in range(num_requests):
        start = time.time()
        response = client.post("/predict", json=request)
        elapsed = (time.time() - start) * 1000
        latencies.append(elapsed)
        assert response.status_code == 200

    import numpy as np

    latencies = np.array(latencies)

    print(f"\nRequests: {num_requests}")
    print(f"Average Latency: {np.mean(latencies):.2f}ms")
    print(f"Median Latency: {np.median(latencies):.2f}ms")
    print(f"P95 Latency: {np.percentile(latencies, 95):.2f}ms")
    print(f"P99 Latency: {np.percentile(latencies, 99):.2f}ms")
    print(f"Min Latency: {np.min(latencies):.2f}ms")
    print(f"Max Latency: {np.max(latencies):.2f}ms")
    print(f"Throughput: {num_requests / (sum(latencies)/1000):.1f} req/sec")
    print("=" * 80)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
