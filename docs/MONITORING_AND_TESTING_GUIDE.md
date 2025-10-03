# Monitoring and Load Testing Guide
## QuantumFlow - Production Observability & Performance Testing

This guide covers setting up comprehensive monitoring with Prometheus + Grafana and load testing with Locust for the QuantumFlow HFT Prediction Engine.

---

## Table of Contents

1. [Overview](#overview)
2. [API Integration Tests](#api-integration-tests)
3. [Prometheus Metrics](#prometheus-metrics)
4. [Grafana Dashboards](#grafana-dashboards)
5. [Load Testing with Locust](#load-testing-with-locust)
6. [Docker Compose Setup](#docker-compose-setup)
7. [Alerting](#alerting)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### Monitoring Stack

```
┌─────────────────────────────────────────────────────────┐
│                     Monitoring Stack                     │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ↓                   ↓                   ↓
┌──────────────┐    ┌──────────────┐   ┌──────────────┐
│  HFT API     │    │  Prometheus  │   │   Grafana    │
│              │───→│              │──→│              │
│ (metrics     │    │ (collection) │   │ (visualization)
│  exporter)   │    │              │   │              │
└──────────────┘    └──────────────┘   └──────────────┘
                            │
                            ↓
                    ┌──────────────┐
                    │ Alertmanager │
                    │ (alerts)     │
                    └──────────────┘
```

### What We Monitor

**API Performance:**
- Request rate (req/sec)
- Latency (p50, p95, p99)
- Error rate by type
- Active requests

**ML Models:**
- Prediction count by model
- Prediction latency
- Confidence distribution
- Model accuracy

**System Resources:**
- CPU usage
- Memory usage
- Cache hit rate
- Cache size

**Business Metrics:**
- Snapshots processed
- Features computed
- Predictions by class

---

## API Integration Tests

### Running Tests

```bash
# Run all API tests
pytest tests/test_api_integration.py -v

# Run specific test class
pytest tests/test_api_integration.py::TestAPIPredictionEndpoint -v

# Run with coverage
pytest tests/test_api_integration.py -v --cov=src/api

# Run performance benchmark
pytest tests/test_api_integration.py::test_api_benchmark -v -s
```

### Test Coverage

Our API integration tests cover:

✅ **Health Endpoints** (3 tests)
- Root endpoint
- Health check
- Models listing

✅ **Prediction Endpoint** (12 tests)
- Valid requests
- Cache functionality
- Invalid model names
- Empty snapshots
- Size limits (max 1000)
- Invalid structure
- Negative prices
- Crossed books
- Invalid exchange/symbol names
- Feature inclusion

✅ **Rate Limiting** (1 test)
- Enforcement of 10 req/min limit

✅ **Metrics** (3 tests)
- Metrics structure
- Metrics updates
- Cache tracking
- Metrics reset

✅ **Performance** (3 tests)
- Latency thresholds
- Throughput capacity
- Large request handling

✅ **Error Handling** (4 tests)
- Invalid JSON
- Missing fields
- Wrong data types
- Wrong HTTP methods

✅ **Documentation** (3 tests)
- OpenAPI docs
- ReDoc
- JSON schema

**Total: 29 test methods**

### Example Test Output

```
tests/test_api_integration.py::TestAPIHealthEndpoints::test_health_check_endpoint PASSED
tests/test_api_integration.py::TestAPIPredictionEndpoint::test_valid_prediction_request PASSED
tests/test_api_integration.py::TestAPIPredictionEndpoint::test_cache_functionality PASSED
tests/test_api_integration.py::TestAPIRateLimiting::test_rate_limit_enforcement PASSED
tests/test_api_integration.py::TestAPIPerformance::test_prediction_latency_under_threshold PASSED

================================ 29 passed in 3.45s ================================
```

---

## Prometheus Metrics

### Available Metrics

#### System Info
```
hft_api_info{version="2.0.0", environment="production"}
```

#### Request Metrics
```
# Total requests
hft_api_requests_total{endpoint="/predict", method="POST", status="200"}

# Request latency histogram
hft_api_request_latency_seconds_bucket{endpoint="/predict", le="0.05"}

# Active requests gauge
hft_api_active_requests{endpoint="/predict"}
```

#### Prediction Metrics
```
# Total predictions
hft_predictions_total{model_name="lstm_v1", prediction_class="up"}

# Prediction latency
hft_prediction_latency_seconds_bucket{model_name="lstm_v1", le="0.05"}

# Prediction confidence
hft_prediction_confidence_bucket{model_name="lstm_v1", prediction_class="up", le="0.7"}
```

#### Cache Metrics
```
# Cache hits/misses
hft_cache_hits_total
hft_cache_misses_total

# Cache size
hft_cache_size_bytes
```

#### Error Metrics
```
# Total errors
hft_errors_total{error_type="ValueError", endpoint="/predict"}

# Validation errors
hft_validation_errors_total{field="snapshots", error_type="empty"}
```

#### Business Metrics
```
# Snapshots processed
hft_snapshots_processed_total{exchange="binance", symbol="BTCUSDT"}

# Features computed
hft_features_computed_total{feature_type="ofi"}

# Model accuracy
hft_model_accuracy{model_name="lstm_v1"}
```

### Using Prometheus Exporter

```python
from src.monitoring import PrometheusMetrics

# Initialize metrics
metrics = PrometheusMetrics()

# Track request
with metrics.track_request("/predict", "POST"):
    # Handle request
    pass

# Track prediction
with metrics.track_prediction_latency("lstm_v1"):
    # Make prediction
    result = model.predict(x)

# Record prediction
metrics.record_prediction(
    model_name="lstm_v1",
    prediction_class="up",
    confidence=0.75,
    latency_ms=23.5
)

# Record cache
metrics.record_cache_hit()

# Record error
metrics.record_error("ValueError", "/predict")

# Export metrics for Prometheus
metrics_output = metrics.generate_metrics()
```

### Querying Metrics

```bash
# Get current metrics
curl http://localhost:8000/metrics

# Example queries in Prometheus
# Request rate
rate(hft_api_requests_total[5m])

# P95 latency
histogram_quantile(0.95, rate(hft_api_request_latency_seconds_bucket[5m]))

# Cache hit rate
rate(hft_cache_hits_total[5m]) / (rate(hft_cache_hits_total[5m]) + rate(hft_cache_misses_total[5m]))

# Error rate
rate(hft_errors_total[5m])
```

---

## Grafana Dashboards

### Dashboard Overview

Our Grafana dashboard includes **15 panels**:

1. **API Request Rate** - Requests per second by endpoint
2. **API Latency (P95)** - P50 and P95 latency over time
3. **Active Requests** - Currently processing requests
4. **Total Predictions** - Cumulative prediction count
5. **Cache Hit Rate** - Percentage of cache hits (gauge)
6. **Error Rate** - Errors per second by type
7. **Prediction Latency by Model** - Heatmap of latencies
8. **Predictions by Model and Class** - Pie chart distribution
9. **Model Accuracy** - Accuracy over time per model
10. **Prediction Confidence Distribution** - Histogram
11. **Snapshots Processed** - By exchange and symbol
12. **Features Computed** - By feature type
13. **Memory Usage** - Memory consumption over time
14. **CPU Usage** - CPU percentage over time
15. **Cache Size** - Cache size in bytes

### Importing Dashboard

```bash
# 1. Start Grafana
docker-compose up -d grafana

# 2. Access Grafana
open http://localhost:3000
# Default login: admin / admin

# 3. Add Prometheus data source
# Go to: Configuration → Data Sources → Add data source
# Select: Prometheus
# URL: http://prometheus:9090
# Click: Save & Test

# 4. Import dashboard
# Go to: Dashboards → Import
# Upload: monitoring/grafana/hft-api-dashboard.json
```

### Dashboard Alerts

Built-in alerts in the dashboard:

- **High API Latency** - P95 > 100ms for 2 minutes
- **High Error Rate** - > 1 error/sec for 2 minutes
- **Low Model Accuracy** - < 50% for 5 minutes

---

## Load Testing with Locust

### Installation

```bash
pip install locust
```

### Running Load Tests

#### 1. Basic Load Test (Web UI)

```bash
locust -f tests/load_testing/locustfile.py --host=http://localhost:8000

# Open browser to http://localhost:8089
# Configure users and spawn rate in UI
```

#### 2. Headless Mode (Automated)

```bash
# Standard load test
locust -f tests/load_testing/locustfile.py \
       --host=http://localhost:8000 \
       --users 50 \
       --spawn-rate 5 \
       --run-time 5m \
       --headless

# Stress test (aggressive)
locust -f tests/load_testing/locustfile.py \
       --host=http://localhost:8000 \
       --users 200 \
       --spawn-rate 20 \
       --user-classes StressTestUser \
       --run-time 3m \
       --headless

# Realistic trading simulation
locust -f tests/load_testing/locustfile.py \
       --host=http://localhost:8000 \
       --users 30 \
       --spawn-rate 3 \
       --user-classes RealisticTradingUser \
       --run-time 10m \
       --headless
```

#### 3. Advanced Load Patterns

```bash
# Step load (gradually increase)
locust -f tests/load_testing/locustfile.py \
       --host=http://localhost:8000 \
       --shape StepLoadShape \
       --headless

# Spike test (sudden traffic bursts)
locust -f tests/load_testing/locustfile.py \
       --host=http://localhost:8000 \
       --shape SpikeLoadShape \
       --headless
```

### Load Test Scenarios

**HFTAPIUser (Default):**
- Simulates normal API usage
- Mixed batch sizes (small/medium/large)
- Health checks and metrics queries
- Tests error handling

**StressTestUser:**
- Rapid-fire requests (0.1-0.5s wait)
- Minimal payloads for maximum throughput
- Tests system limits

**RealisticTradingUser:**
- Simulates real trader behavior
- Decision-making delays (5-15s)
- Confidence-based actions
- Periodic system checks

**StepLoadShape:**
- Gradually increases load
- 10 users per step
- 60 seconds per step
- Finds breaking points

**SpikeLoadShape:**
- Normal load: 20 users
- Spike load: 100 users
- Spikes every 2 minutes for 30 seconds
- Tests resilience

### Interpreting Results

#### Good Performance
```
Type     Name                          # reqs    # fails  Avg (ms)  P95 (ms)  P99 (ms)  RPS
--------  -------------------------------  -------  --------  --------  --------  --------  ---
POST      /predict (small)              5000     0         45        78        95        167
POST      /predict (medium)             2500     0         67        112       145       83
POST      /predict (large)              1000     0         89        156       189       33
GET       /health                       1500     0         5         8         12        50
--------  -------------------------------  -------  --------  --------  --------  --------  ---
Total                                    10000    0         52        98        142       333

✅ Failure rate: 0%
✅ P95 latency: < 100ms (small), < 200ms (large)
✅ RPS: 333 requests/sec
```

#### Warning Signs
```
Type     Name                          # reqs    # fails  Avg (ms)  P95 (ms)  P99 (ms)  RPS
--------  -------------------------------  -------  --------  --------  --------  --------  ---
POST      /predict (small)              5000     250       523       890       1200      125
...
--------  -------------------------------  -------  --------  --------  --------  --------  ---
Total                                    10000    500       478       825       1150      250

⚠️  Failure rate: 5%
⚠️  P95 latency: > 500ms
⚠️  Increasing latency trend
```

#### Action Needed
```
Type     Name                          # reqs    # fails  Avg (ms)  P95 (ms)  P99 (ms)  RPS
--------  -------------------------------  -------  --------  --------  --------  --------  ---
POST      /predict (small)              5000     1200      1245      2300      3400      83
...
--------  -------------------------------  -------  --------  --------  --------  --------  ---
Total                                    10000    2500      1134      2150      3200      167

🚨 Failure rate: 25%
🚨 P95 latency: > 2000ms
🚨 RPS declining over time
```

---

## Docker Compose Setup

### Full Monitoring Stack

Create `docker-compose.monitoring.yml`:

```yaml
version: '3.8'

services:
  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    container_name: hft-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/prometheus/alerts.yml:/etc/prometheus/alerts.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
    networks:
      - monitoring

  # Grafana
  grafana:
    image: grafana/grafana:latest
    container_name: hft-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - ./monitoring/grafana:/etc/grafana/provisioning
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
    networks:
      - monitoring

  # Alertmanager
  alertmanager:
    image: prom/alertmanager:latest
    container_name: hft-alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
    networks:
      - monitoring

  # Node Exporter (system metrics)
  node-exporter:
    image: prom/node-exporter:latest
    container_name: hft-node-exporter
    ports:
      - "9100:9100"
    networks:
      - monitoring

volumes:
  prometheus_data:
  grafana_data:

networks:
  monitoring:
    driver: bridge
```

### Starting the Stack

```bash
# Start monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# View logs
docker-compose -f docker-compose.monitoring.yml logs -f

# Stop stack
docker-compose -f docker-compose.monitoring.yml down
```

### Accessing Services

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Alertmanager**: http://localhost:9093

---

## Alerting

### Alert Rules

We have 11 alert rules defined:

1. **HighAPILatency** - P95 > 100ms for 2 min (warning)
2. **CriticalAPILatency** - P95 > 500ms for 1 min (critical)
3. **HighErrorRate** - > 1 error/sec for 2 min (warning)
4. **CriticalErrorRate** - > 5 errors/sec for 1 min (critical)
5. **LowCacheHitRate** - < 30% for 5 min (warning)
6. **LowModelAccuracy** - < 50% for 5 min (warning)
7. **HighMemoryUsage** - > 8GB for 5 min (warning)
8. **HighCPUUsage** - > 80% for 5 min (warning)
9. **APIDown** - Service down for 1 min (critical)
10. **LowRequestRate** - < 0.1 req/sec for 10 min (info)
11. **PredictionLatencyAnomaly** - 2x normal for 3 min (warning)

### Configuring Alertmanager

Create `monitoring/alertmanager/alertmanager.yml`:

```yaml
global:
  slack_api_url: 'YOUR_SLACK_WEBHOOK_URL'

route:
  receiver: 'default'
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h

  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
    - match:
        severity: warning
      receiver: 'warning-alerts'

receivers:
  - name: 'default'
    email_configs:
      - to: 'alerts@yourdomain.com'
        from: 'prometheus@yourdomain.com'
        smarthost: 'smtp.gmail.com:587'
        auth_username: 'your-email@gmail.com'
        auth_password: 'your-password'

  - name: 'critical-alerts'
    slack_configs:
      - channel: '#critical-alerts'
        title: '🚨 Critical Alert'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

  - name: 'warning-alerts'
    slack_configs:
      - channel: '#monitoring'
        title: '⚠️  Warning'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
```

---

## Troubleshooting

### Common Issues

#### 1. Metrics Not Appearing in Prometheus

**Problem:** Prometheus shows no metrics from API

**Solution:**
```bash
# Check API is exposing metrics
curl http://localhost:8000/metrics

# Check Prometheus config
docker exec hft-prometheus cat /etc/prometheus/prometheus.yml

# Check Prometheus targets
# Go to http://localhost:9090/targets
# Ensure hft-api is "UP"

# Restart Prometheus
docker-compose -f docker-compose.monitoring.yml restart prometheus
```

#### 2. Grafana Can't Connect to Prometheus

**Problem:** "Bad Gateway" or "Connection refused"

**Solution:**
```bash
# Use Docker network name, not localhost
# In Grafana datasource, use: http://prometheus:9090

# Check networks
docker network inspect monitoring

# Ensure both containers are on same network
docker-compose -f docker-compose.monitoring.yml ps
```

#### 3. High Test Failure Rate

**Problem:** Load tests show >10% failures

**Solution:**
```bash
# Check API logs
docker-compose logs -f hft-api

# Check resource usage
docker stats

# Reduce load
locust -f locustfile.py --users 20 --spawn-rate 2

# Increase API resources
# Edit docker-compose.yml:
services:
  hft-api:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
```

#### 4. Rate Limiting Causing Failures

**Problem:** Many 429 errors in tests

**Solution:**
```python
# This is expected behavior!
# Rate limit is 10 req/min per IP

# In load tests, check for 429 specifically:
if response.status_code == 429:
    response.success()  # Expected, not a failure

# Or increase rate limit in API config
```

---

## Best Practices

### 1. Monitoring

✅ **DO:**
- Monitor business metrics (predictions, accuracy)
- Set up alerts for critical paths
- Review dashboards regularly
- Track trends over time
- Monitor dependencies (DB, Redis, Kafka)

❌ **DON'T:**
- Alert on everything (too noisy)
- Ignore warning alerts
- Set thresholds too tight
- Forget to test alerts

### 2. Load Testing

✅ **DO:**
- Test before production
- Use realistic data
- Test different scenarios
- Monitor during tests
- Document breaking points

❌ **DON'T:**
- Test on production
- Use only small payloads
- Ignore resource limits
- Skip spike testing
- Forget to baseline

### 3. Performance Tuning

Based on test results:

**If latency is high:**
- Add caching
- Optimize database queries
- Use connection pooling
- Scale horizontally

**If error rate is high:**
- Add retry logic
- Improve error handling
- Add circuit breakers
- Validate inputs early

**If RPS is low:**
- Use async/await
- Batch processing
- Reduce serialization overhead
- Profile code

---

## Summary

You now have:

✅ **29 API integration tests** - Comprehensive coverage
✅ **20+ Prometheus metrics** - Full observability
✅ **15-panel Grafana dashboard** - Beautiful visualization
✅ **11 alert rules** - Proactive monitoring
✅ **3 Locust scenarios** - Realistic load testing
✅ **Docker Compose setup** - Easy deployment

**Next Steps:**

1. Run API tests: `pytest tests/test_api_integration.py -v`
2. Start monitoring: `docker-compose -f docker-compose.monitoring.yml up -d`
3. Import Grafana dashboard
4. Run load test: `locust -f tests/load_testing/locustfile.py`
5. Set up alerts in your notification system

---

**Your API is now production-ready with enterprise-grade monitoring!** 🚀
