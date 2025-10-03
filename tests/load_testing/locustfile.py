"""
Locust Load Testing for HFT Prediction API

This file defines load testing scenarios using Locust.

Run with:
    # Basic load test
    locust -f tests/load_testing/locustfile.py --host=http://localhost:8000

    # Headless mode with specific users
    locust -f tests/load_testing/locustfile.py --host=http://localhost:8000 \
           --users 100 --spawn-rate 10 --run-time 5m --headless

    # Web UI mode
    locust -f tests/load_testing/locustfile.py --host=http://localhost:8000 --web-host=0.0.0.0
"""

import random
import time
from locust import HttpUser, task, between, events
from locust.contrib.fasthttp import FastHttpUser


class HFTAPIUser(FastHttpUser):
    """
    Simulates a user making requests to the HFT API.

    Uses FastHttpUser for better performance (gevent-based).
    """

    # Wait between 1-3 seconds between tasks
    wait_time = between(1, 3)

    # Sample order book data
    def get_sample_snapshot(self):
        """Generate realistic order book snapshot."""
        base_price = 50000 + random.uniform(-1000, 1000)
        spread = random.uniform(5, 20)

        return {
            "timestamp": int(time.time() * 1000),
            "exchange": random.choice(["binance", "coinbase", "kraken"]),
            "symbol": random.choice(["BTCUSDT", "ETHUSDT", "BNBUSDT"]),
            "bids": [
                [base_price - spread - i * 1, random.uniform(0.5, 5.0)]
                for i in range(10)
            ],
            "asks": [[base_price + i * 1, random.uniform(0.5, 5.0)] for i in range(10)],
            "sequence": random.randint(10000, 99999),
        }

    def on_start(self):
        """Called when a simulated user starts."""
        # Could do authentication here if needed
        pass

    @task(10)  # Weight: 10 (most common)
    def predict_small_batch(self):
        """Make prediction with small batch (1-10 snapshots)."""
        num_snapshots = random.randint(1, 10)
        payload = {
            "snapshots": [self.get_sample_snapshot() for _ in range(num_snapshots)],
            "model_name": "dummy",
            "include_features": False,
        }

        with self.client.post(
            "/predict", json=payload, catch_response=True, name="/predict (small)"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                # Validate response
                if "prediction" in data and "confidence" in data:
                    response.success()
                else:
                    response.failure("Missing required fields in response")
            elif response.status_code == 429:
                # Rate limited - expected behavior
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")

    @task(5)  # Weight: 5 (medium frequency)
    def predict_medium_batch(self):
        """Make prediction with medium batch (10-50 snapshots)."""
        num_snapshots = random.randint(10, 50)
        payload = {
            "snapshots": [self.get_sample_snapshot() for _ in range(num_snapshots)],
            "model_name": "dummy",
            "include_features": False,
        }

        with self.client.post(
            "/predict", json=payload, catch_response=True, name="/predict (medium)"
        ) as response:
            if response.status_code in [200, 429]:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(2)  # Weight: 2 (less frequent)
    def predict_large_batch(self):
        """Make prediction with large batch (50-100 snapshots)."""
        num_snapshots = random.randint(50, 100)
        payload = {
            "snapshots": [self.get_sample_snapshot() for _ in range(num_snapshots)],
            "model_name": "dummy",
            "include_features": False,
        }

        with self.client.post(
            "/predict", json=payload, catch_response=True, name="/predict (large)"
        ) as response:
            if response.status_code in [200, 429]:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(3)  # Weight: 3
    def check_health(self):
        """Check API health."""
        with self.client.get(
            "/health", name="/health", catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure("API not healthy")
            else:
                response.failure(f"Status: {response.status_code}")

    @task(2)  # Weight: 2
    def get_metrics(self):
        """Get API metrics."""
        with self.client.get(
            "/metrics", name="/metrics", catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(2)  # Weight: 2
    def list_models(self):
        """List available models."""
        with self.client.get(
            "/models", name="/models", catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "models" in data:
                    response.success()
                else:
                    response.failure("Missing models field")
            else:
                response.failure(f"Status: {response.status_code}")

    @task(1)  # Weight: 1 (rare)
    def test_invalid_request(self):
        """Test API error handling with invalid request."""
        payload = {
            "snapshots": [],  # Empty (should fail validation)
            "model_name": "dummy",
        }

        with self.client.post(
            "/predict", json=payload, catch_response=True, name="/predict (invalid)"
        ) as response:
            # Should return validation error (422)
            if response.status_code == 422:
                response.success()
            else:
                response.failure(f"Expected 422, got {response.status_code}")


class StressTestUser(FastHttpUser):
    """
    Aggressive stress testing user.

    Sends requests as fast as possible to test limits.
    """

    wait_time = between(0.1, 0.5)  # Very short wait time

    def get_sample_snapshot(self):
        """Generate minimal snapshot for speed."""
        return {
            "timestamp": int(time.time() * 1000),
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "bids": [[50000.0, 1.0]],
            "asks": [[50001.0, 1.0]],
        }

    @task
    def rapid_fire_predictions(self):
        """Send predictions rapidly."""
        payload = {"snapshots": [self.get_sample_snapshot()], "model_name": "dummy"}

        self.client.post("/predict", json=payload, name="/predict (stress)")


class RealisticTradingUser(HttpUser):
    """
    Simulates realistic trading patterns.

    - Checks health periodically
    - Makes predictions based on market activity
    - Simulates decision-making delays
    """

    wait_time = between(5, 15)  # Realistic thinking time

    def get_sample_snapshot(self):
        """Generate realistic order book snapshot."""
        base_price = 50000
        return {
            "timestamp": int(time.time() * 1000),
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "bids": [[base_price - i, random.uniform(0.1, 2.0)] for i in range(20)],
            "asks": [[base_price + i + 5, random.uniform(0.1, 2.0)] for i in range(20)],
        }

    @task(15)
    def analyze_market(self):
        """Analyze market and make prediction."""
        # Simulate collecting recent snapshots
        num_snapshots = random.randint(20, 50)
        payload = {
            "snapshots": [self.get_sample_snapshot() for _ in range(num_snapshots)],
            "model_name": "dummy",
            "include_features": False,
        }

        response = self.client.post("/predict", json=payload)

        if response.status_code == 200:
            data = response.json()
            confidence = data.get("confidence", 0)

            # Simulate decision based on confidence
            if confidence > 0.7:
                # High confidence - would execute trade
                time.sleep(random.uniform(0.5, 1.5))
            else:
                # Low confidence - wait and observe
                time.sleep(random.uniform(2, 5))

    @task(5)
    def check_system_status(self):
        """Periodically check system health."""
        self.client.get("/health")
        time.sleep(1)
        self.client.get("/metrics")


# ============================================================================
# Event Handlers for Custom Reporting
# ============================================================================


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts."""
    print("\n" + "=" * 80)
    print("LOAD TEST STARTING")
    print("=" * 80)
    print(f"Target: {environment.host}")
    print(
        f"Users: {environment.runner.target_user_count if hasattr(environment.runner, 'target_user_count') else 'N/A'}"
    )
    print("=" * 80 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops."""
    print("\n" + "=" * 80)
    print("LOAD TEST COMPLETED")
    print("=" * 80)

    stats = environment.stats

    print("\nSummary:")
    print(f"  Total Requests: {stats.total.num_requests}")
    print(f"  Total Failures: {stats.total.num_failures}")
    print(f"  Failure Rate: {stats.total.fail_ratio * 100:.2f}%")
    print(f"  Average Response Time: {stats.total.avg_response_time:.2f}ms")
    print(f"  Median Response Time: {stats.total.median_response_time:.2f}ms")
    print(f"  95th Percentile: {stats.total.get_response_time_percentile(0.95):.2f}ms")
    print(f"  99th Percentile: {stats.total.get_response_time_percentile(0.99):.2f}ms")
    print(f"  Max Response Time: {stats.total.max_response_time:.2f}ms")
    print(f"  Requests/sec: {stats.total.total_rps:.2f}")

    print("\n" + "=" * 80 + "\n")


# ============================================================================
# Custom Shape for Advanced Load Patterns
# ============================================================================

from locust import LoadTestShape


class StepLoadShape(LoadTestShape):
    """
    A step load shape that gradually increases load.

    Useful for finding breaking points.
    """

    step_time = 60  # Each step lasts 60 seconds
    step_load = 10  # Increase by 10 users each step
    spawn_rate = 5
    time_limit = 600  # Total test duration: 10 minutes

    def tick(self):
        """Return user count and spawn rate for current time."""
        run_time = self.get_run_time()

        if run_time > self.time_limit:
            return None

        current_step = run_time // self.step_time
        return (current_step + 1) * self.step_load, self.spawn_rate


class SpikeLoadShape(LoadTestShape):
    """
    Simulates traffic spikes.

    Normal load with occasional spikes to test resilience.
    """

    time_limit = 600
    normal_users = 20
    spike_users = 100
    spawn_rate = 20

    def tick(self):
        """Return user count and spawn rate for current time."""
        run_time = self.get_run_time()

        if run_time > self.time_limit:
            return None

        # Spike every 2 minutes for 30 seconds
        cycle_time = run_time % 120
        if cycle_time < 30:
            # Spike!
            return self.spike_users, self.spawn_rate
        else:
            # Normal load
            return self.normal_users, self.spawn_rate


# ============================================================================
# Usage Instructions
# ============================================================================

"""
USAGE EXAMPLES:

1. Basic load test with web UI:
   locust -f tests/load_testing/locustfile.py --host=http://localhost:8000

2. Headless mode (no UI):
   locust -f tests/load_testing/locustfile.py --host=http://localhost:8000 \
          --users 50 --spawn-rate 5 --run-time 5m --headless

3. Stress test:
   locust -f tests/load_testing/locustfile.py --host=http://localhost:8000 \
          --users 200 --spawn-rate 20 -H StressTestUser --headless

4. Realistic trading simulation:
   locust -f tests/load_testing/locustfile.py --host=http://localhost:8000 \
          --users 30 --spawn-rate 3 -H RealisticTradingUser --headless

5. Step load (find breaking point):
   locust -f tests/load_testing/locustfile.py --host=http://localhost:8000 \
          --shape=StepLoadShape --headless

6. Spike test:
   locust -f tests/load_testing/locustfile.py --host=http://localhost:8000 \
          --shape=SpikeLoadShape --headless

INTERPRETING RESULTS:

Good Performance:
  - P95 latency < 100ms
  - Failure rate < 1%
  - RPS > 100

Warning Signs:
  - P95 latency > 500ms
  - Failure rate > 5%
  - Increasing latency over time

Action Needed:
  - P95 latency > 1000ms
  - Failure rate > 10%
  - RPS declining over time
"""
