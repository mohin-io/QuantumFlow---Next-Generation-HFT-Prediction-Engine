"""
Prometheus Metrics Exporter for HFT API

Exposes metrics in Prometheus format for monitoring with Grafana.

Features:
- Request count and rate
- Latency histograms (p50, p95, p99)
- Error rates by type
- Cache hit rates
- Model usage statistics
- Custom business metrics

Usage:
    from src.monitoring.prometheus_exporter import PrometheusMetrics

    metrics = PrometheusMetrics()

    # Record prediction
    with metrics.track_prediction_latency(model_name="lstm_v1"):
        # Make prediction
        pass

    # Export metrics
    metrics_output = metrics.generate_metrics()
"""

import time
from collections import defaultdict
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    CollectorRegistry,
    CONTENT_TYPE_LATEST
)


class PrometheusMetrics:
    """Prometheus metrics collector for HFT API."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize Prometheus metrics.

        Args:
            registry: Custom registry (default: creates new one)
        """
        self.registry = registry or CollectorRegistry()
        self.lock = Lock()

        # ================================================================
        # System Info
        # ================================================================
        self.system_info = Info(
            'hft_api_info',
            'HFT API system information',
            registry=self.registry
        )
        self.system_info.info({
            'version': '2.0.0',
            'environment': 'production',
            'service': 'hft-prediction-api'
        })

        # ================================================================
        # Request Metrics
        # ================================================================
        self.request_count = Counter(
            'hft_api_requests_total',
            'Total number of API requests',
            ['endpoint', 'method', 'status'],
            registry=self.registry
        )

        self.request_latency = Histogram(
            'hft_api_request_latency_seconds',
            'Request latency in seconds',
            ['endpoint', 'method'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            registry=self.registry
        )

        self.active_requests = Gauge(
            'hft_api_active_requests',
            'Number of active requests',
            ['endpoint'],
            registry=self.registry
        )

        # ================================================================
        # Prediction Metrics
        # ================================================================
        self.predictions_total = Counter(
            'hft_predictions_total',
            'Total number of predictions made',
            ['model_name', 'prediction_class'],
            registry=self.registry
        )

        self.prediction_latency = Histogram(
            'hft_prediction_latency_seconds',
            'Prediction latency in seconds',
            ['model_name'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
            registry=self.registry
        )

        self.prediction_confidence = Histogram(
            'hft_prediction_confidence',
            'Prediction confidence scores',
            ['model_name', 'prediction_class'],
            buckets=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
            registry=self.registry
        )

        # ================================================================
        # Cache Metrics
        # ================================================================
        self.cache_hits = Counter(
            'hft_cache_hits_total',
            'Total cache hits',
            registry=self.registry
        )

        self.cache_misses = Counter(
            'hft_cache_misses_total',
            'Total cache misses',
            registry=self.registry
        )

        self.cache_size = Gauge(
            'hft_cache_size_bytes',
            'Current cache size in bytes',
            registry=self.registry
        )

        # ================================================================
        # Error Metrics
        # ================================================================
        self.errors_total = Counter(
            'hft_errors_total',
            'Total errors',
            ['error_type', 'endpoint'],
            registry=self.registry
        )

        self.validation_errors = Counter(
            'hft_validation_errors_total',
            'Total validation errors',
            ['field', 'error_type'],
            registry=self.registry
        )

        # ================================================================
        # Business Metrics
        # ================================================================
        self.snapshots_processed = Counter(
            'hft_snapshots_processed_total',
            'Total order book snapshots processed',
            ['exchange', 'symbol'],
            registry=self.registry
        )

        self.features_computed = Counter(
            'hft_features_computed_total',
            'Total features computed',
            ['feature_type'],
            registry=self.registry
        )

        # ================================================================
        # Model Performance Metrics
        # ================================================================
        self.model_accuracy = Gauge(
            'hft_model_accuracy',
            'Model accuracy (recent window)',
            ['model_name'],
            registry=self.registry
        )

        self.model_prediction_distribution = Counter(
            'hft_model_predictions_by_class',
            'Distribution of predictions by class',
            ['model_name', 'predicted_class'],
            registry=self.registry
        )

        # ================================================================
        # Resource Metrics
        # ================================================================
        self.memory_usage = Gauge(
            'hft_memory_usage_bytes',
            'Current memory usage in bytes',
            registry=self.registry
        )

        self.cpu_usage_percent = Gauge(
            'hft_cpu_usage_percent',
            'Current CPU usage percentage',
            registry=self.registry
        )

    # ====================================================================
    # Context Managers for Automatic Tracking
    # ====================================================================

    def track_request(self, endpoint: str, method: str = "POST"):
        """
        Context manager to track request metrics.

        Usage:
            with metrics.track_request("/predict", "POST"):
                # Handle request
                pass
        """
        class RequestTracker:
            def __init__(self, parent, endpoint, method):
                self.parent = parent
                self.endpoint = endpoint
                self.method = method
                self.start_time = None

            def __enter__(self):
                self.start_time = time.time()
                self.parent.active_requests.labels(endpoint=self.endpoint).inc()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                self.parent.active_requests.labels(endpoint=self.endpoint).dec()

                status = "500" if exc_type else "200"
                self.parent.request_count.labels(
                    endpoint=self.endpoint,
                    method=self.method,
                    status=status
                ).inc()

                self.parent.request_latency.labels(
                    endpoint=self.endpoint,
                    method=self.method
                ).observe(duration)

        return RequestTracker(self, endpoint, method)

    def track_prediction_latency(self, model_name: str):
        """
        Context manager to track prediction latency.

        Usage:
            with metrics.track_prediction_latency("lstm_v1"):
                # Make prediction
                result = model.predict(x)
        """
        class PredictionTracker:
            def __init__(self, parent, model_name):
                self.parent = parent
                self.model_name = model_name
                self.start_time = None

            def __enter__(self):
                self.start_time = time.time()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                self.parent.prediction_latency.labels(
                    model_name=self.model_name
                ).observe(duration)

        return PredictionTracker(self, model_name)

    # ====================================================================
    # Recording Methods
    # ====================================================================

    def record_prediction(
        self,
        model_name: str,
        prediction_class: str,
        confidence: float,
        latency_ms: float
    ):
        """
        Record a prediction event.

        Args:
            model_name: Name of the model
            prediction_class: Predicted class (up/down/flat)
            confidence: Confidence score (0-1)
            latency_ms: Prediction latency in milliseconds
        """
        self.predictions_total.labels(
            model_name=model_name,
            prediction_class=prediction_class
        ).inc()

        self.prediction_confidence.labels(
            model_name=model_name,
            prediction_class=prediction_class
        ).observe(confidence)

        self.prediction_latency.labels(
            model_name=model_name
        ).observe(latency_ms / 1000.0)

        self.model_prediction_distribution.labels(
            model_name=model_name,
            predicted_class=prediction_class
        ).inc()

    def record_cache_hit(self):
        """Record a cache hit."""
        self.cache_hits.inc()

    def record_cache_miss(self):
        """Record a cache miss."""
        self.cache_misses.inc()

    def record_error(self, error_type: str, endpoint: str):
        """
        Record an error.

        Args:
            error_type: Type of error (ValueError, TimeoutError, etc.)
            endpoint: API endpoint where error occurred
        """
        self.errors_total.labels(
            error_type=error_type,
            endpoint=endpoint
        ).inc()

    def record_validation_error(self, field: str, error_type: str):
        """
        Record a validation error.

        Args:
            field: Field that failed validation
            error_type: Type of validation error
        """
        self.validation_errors.labels(
            field=field,
            error_type=error_type
        ).inc()

    def record_snapshots_processed(self, exchange: str, symbol: str, count: int = 1):
        """
        Record processed order book snapshots.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            count: Number of snapshots processed
        """
        self.snapshots_processed.labels(
            exchange=exchange,
            symbol=symbol
        ).inc(count)

    def record_feature_computation(self, feature_type: str):
        """
        Record feature computation.

        Args:
            feature_type: Type of feature (ofi, micro_price, etc.)
        """
        self.features_computed.labels(feature_type=feature_type).inc()

    def update_model_accuracy(self, model_name: str, accuracy: float):
        """
        Update model accuracy metric.

        Args:
            model_name: Name of the model
            accuracy: Accuracy value (0-1)
        """
        self.model_accuracy.labels(model_name=model_name).set(accuracy)

    def update_resource_usage(self, memory_bytes: int, cpu_percent: float):
        """
        Update resource usage metrics.

        Args:
            memory_bytes: Memory usage in bytes
            cpu_percent: CPU usage percentage (0-100)
        """
        self.memory_usage.set(memory_bytes)
        self.cpu_usage_percent.set(cpu_percent)

    def update_cache_size(self, size_bytes: int):
        """
        Update cache size metric.

        Args:
            size_bytes: Cache size in bytes
        """
        self.cache_size.set(size_bytes)

    # ====================================================================
    # Export Methods
    # ====================================================================

    def generate_metrics(self) -> bytes:
        """
        Generate Prometheus metrics in text format.

        Returns:
            Metrics in Prometheus exposition format
        """
        return generate_latest(self.registry)

    def get_content_type(self) -> str:
        """Get content type for Prometheus metrics."""
        return CONTENT_TYPE_LATEST


# ========================================================================
# FastAPI Integration
# ========================================================================

def create_prometheus_middleware(metrics: PrometheusMetrics):
    """
    Create FastAPI middleware for automatic metrics collection.

    Args:
        metrics: PrometheusMetrics instance

    Returns:
        Middleware function
    """
    async def prometheus_middleware(request, call_next):
        """Middleware to track all requests."""
        endpoint = request.url.path
        method = request.method

        with metrics.track_request(endpoint, method):
            response = await call_next(request)
            return response

    return prometheus_middleware


# ========================================================================
# Example Usage
# ========================================================================

if __name__ == "__main__":
    import psutil
    import os

    # Create metrics instance
    metrics = PrometheusMetrics()

    print("Simulating API activity...\n")

    # Simulate some predictions
    for i in range(100):
        model = "lstm_v1" if i % 2 == 0 else "transformer_v1"
        pred_class = ["up", "down", "flat"][i % 3]
        confidence = 0.5 + (i % 50) / 100
        latency = 10 + (i % 40)

        metrics.record_prediction(model, pred_class, confidence, latency)

        if i % 3 == 0:
            metrics.record_cache_hit()
        else:
            metrics.record_cache_miss()

        if i % 20 == 0:
            metrics.record_snapshots_processed("binance", "BTCUSDT", 10)

    # Update resource metrics
    process = psutil.Process(os.getpid())
    metrics.update_resource_usage(
        memory_bytes=process.memory_info().rss,
        cpu_percent=process.cpu_percent()
    )

    # Update model accuracy
    metrics.update_model_accuracy("lstm_v1", 0.683)
    metrics.update_model_accuracy("transformer_v1", 0.675)

    # Generate and print metrics
    print("="*80)
    print("PROMETHEUS METRICS OUTPUT")
    print("="*80)
    print(metrics.generate_metrics().decode('utf-8'))
    print("="*80)
    print("\nMetrics generated successfully!")
    print("These can be scraped by Prometheus at /metrics endpoint")
