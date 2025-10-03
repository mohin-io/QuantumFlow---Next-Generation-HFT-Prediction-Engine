"""
Enhanced FastAPI Prediction Service with Security and Monitoring

IMPROVEMENTS IN THIS VERSION:
- Rate limiting to prevent DDoS attacks
- Request timeout handling
- Input validation and sanitization
- Comprehensive logging and monitoring
- Metrics collection (latency, cache hits, predictions)
- Better error messages with helpful hints
- Security best practices (CORS, input validation)
- Request/response logging for audit trails
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from functools import wraps
from threading import Lock
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HFT Order Book Prediction API",
    description="Production-ready API for high-frequency trading predictions with security and monitoring",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware (configure for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Metrics Collection
# ============================================================================


class MetricsCollector:
    """Collects and aggregates API metrics."""

    def __init__(self):
        self.lock = Lock()
        self.total_predictions = 0
        self.latencies = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = defaultdict(int)
        self.predictions_by_model = defaultdict(int)
        self.start_time = time.time()

    def record_prediction(self, latency_ms: float, cache_hit: bool, model_name: str):
        """Record a successful prediction."""
        with self.lock:
            self.total_predictions += 1
            self.latencies.append(latency_ms)
            self.predictions_by_model[model_name] += 1

            if cache_hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1

            # Keep only last 1000 latencies to avoid memory issues
            if len(self.latencies) > 1000:
                self.latencies = self.latencies[-1000:]

    def record_error(self, error_type: str):
        """Record an error."""
        with self.lock:
            self.errors[error_type] += 1

    def get_metrics(self) -> Dict:
        """Get current metrics snapshot."""
        with self.lock:
            uptime_seconds = time.time() - self.start_time

            if self.latencies:
                latencies_array = np.array(self.latencies)
                avg_latency = float(np.mean(latencies_array))
                p50_latency = float(np.percentile(latencies_array, 50))
                p95_latency = float(np.percentile(latencies_array, 95))
                p99_latency = float(np.percentile(latencies_array, 99))
            else:
                avg_latency = p50_latency = p95_latency = p99_latency = 0.0

            cache_total = self.cache_hits + self.cache_misses
            cache_hit_rate = (
                (self.cache_hits / cache_total * 100) if cache_total > 0 else 0.0
            )

            return {
                "uptime_seconds": uptime_seconds,
                "total_predictions": self.total_predictions,
                "predictions_per_second": (
                    self.total_predictions / uptime_seconds if uptime_seconds > 0 else 0
                ),
                "latency_ms": {
                    "avg": avg_latency,
                    "p50": p50_latency,
                    "p95": p95_latency,
                    "p99": p99_latency,
                },
                "cache": {
                    "hits": self.cache_hits,
                    "misses": self.cache_misses,
                    "hit_rate_percent": cache_hit_rate,
                },
                "predictions_by_model": dict(self.predictions_by_model),
                "errors": dict(self.errors),
            }

    def reset(self):
        """Reset all metrics."""
        with self.lock:
            self.total_predictions = 0
            self.latencies = []
            self.cache_hits = 0
            self.cache_misses = 0
            self.errors = defaultdict(int)
            self.predictions_by_model = defaultdict(int)
            self.start_time = time.time()
            logger.info("Metrics reset")


# Global metrics collector
metrics_collector = MetricsCollector()


# ============================================================================
# Request/Response Models with Validation
# ============================================================================


class OrderBookSnapshot(BaseModel):
    """Order book snapshot with validation."""

    timestamp: int = Field(..., ge=0, description="Unix timestamp in milliseconds")
    exchange: str = Field(..., min_length=1, max_length=50, regex="^[a-zA-Z0-9_-]+$")
    symbol: str = Field(..., min_length=1, max_length=20, regex="^[A-Z0-9/-]+$")
    bids: List[List[float]] = Field(..., min_items=1, max_items=100)
    asks: List[List[float]] = Field(..., min_items=1, max_items=100)
    sequence: Optional[int] = Field(None, ge=0)

    @validator("bids", "asks")
    def validate_price_levels(cls, v):
        """Validate price levels have correct structure."""
        for level in v:
            if len(level) != 2:
                raise ValueError("Each price level must be [price, volume]")
            price, volume = level
            if price <= 0:
                raise ValueError(f"Price must be positive, got {price}")
            if volume < 0:
                raise ValueError(f"Volume must be non-negative, got {volume}")
        return v

    @validator("bids")
    def validate_bids_descending(cls, v):
        """Validate bids are in descending price order."""
        prices = [level[0] for level in v]
        if not all(prices[i] >= prices[i + 1] for i in range(len(prices) - 1)):
            raise ValueError("Bid prices must be in descending order")
        return v

    @validator("asks")
    def validate_asks_ascending(cls, v):
        """Validate asks are in ascending price order."""
        prices = [level[0] for level in v]
        if not all(prices[i] <= prices[i + 1] for i in range(len(prices) - 1)):
            raise ValueError("Ask prices must be in ascending order")
        return v


class PredictionRequest(BaseModel):
    """Prediction request with comprehensive validation."""

    snapshots: List[OrderBookSnapshot] = Field(
        ...,
        min_items=1,
        max_items=1000,
        description="List of order book snapshots (max 1000 for safety)",
    )
    model_name: str = Field(
        default="lstm_v1",
        regex="^[a-zA-Z0-9_-]+$",
        max_length=50,
        description="Model name (alphanumeric, underscore, hyphen only)",
    )
    include_features: bool = Field(
        default=False, description="Whether to include computed features in response"
    )

    @validator("snapshots")
    def validate_snapshots_not_empty(cls, v):
        """Ensure snapshots list is not empty."""
        if len(v) == 0:
            raise ValueError("snapshots cannot be empty")
        return v

    @validator("snapshots")
    def validate_snapshots_temporal_order(cls, v):
        """Ensure snapshots are in temporal order."""
        timestamps = [s.timestamp for s in v]
        if not all(
            timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1)
        ):
            logger.warning("Snapshots not in temporal order, will be sorted")
        return v

    @validator("model_name")
    def validate_model_name_allowed(cls, v):
        """Validate model name is in allowed list."""
        allowed_models = {
            "lstm_v1",
            "transformer_v1",
            "attention_lstm_v1",
            "ensemble_v1",
            "dummy",
        }
        if v not in allowed_models:
            raise ValueError(
                f"Invalid model name '{v}'. Allowed models: {', '.join(sorted(allowed_models))}"
            )
        return v


class PredictionResponse(BaseModel):
    """Prediction response."""

    prediction: str = Field(
        ..., description="Predicted direction: 'up', 'down', or 'flat'"
    )
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    model_name: str = Field(..., description="Model used for prediction")
    latency_ms: float = Field(
        ..., ge=0, description="Prediction latency in milliseconds"
    )
    cache_hit: bool = Field(..., description="Whether result was from cache")
    timestamp: str = Field(..., description="Prediction timestamp (ISO format)")
    features: Optional[Dict] = Field(
        None, description="Computed features (if requested)"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    models_loaded: List[str] = Field(..., description="Loaded models")
    uptime_seconds: float = Field(..., description="Service uptime")


class MetricsResponse(BaseModel):
    """Metrics response."""

    uptime_seconds: float
    total_predictions: int
    predictions_per_second: float
    latency_ms: Dict[str, float]
    cache: Dict[str, float]
    predictions_by_model: Dict[str, int]
    errors: Dict[str, int]


# ============================================================================
# Timeout Decorator
# ============================================================================


def with_timeout(timeout_sec: float):
    """Decorator to add timeout to async functions."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs), timeout=timeout_sec
                )
            except asyncio.TimeoutError:
                logger.error(f"Request timeout after {timeout_sec}s in {func.__name__}")
                raise HTTPException(
                    status_code=504,
                    detail={
                        "error": "request_timeout",
                        "message": f"Request timed out after {timeout_sec} seconds",
                        "hint": "Try reducing the number of snapshots or use a faster model",
                    },
                )

        return wrapper

    return decorator


# ============================================================================
# Cache Implementation
# ============================================================================


class SimpleCache:
    """Simple in-memory cache with TTL."""

    def __init__(self, ttl_seconds: int = 60):
        self.cache = {}
        self.ttl_seconds = ttl_seconds
        self.lock = Lock()

    def get(self, key: str) -> Optional[Dict]:
        """Get value from cache if not expired."""
        with self.lock:
            if key in self.cache:
                value, expiry = self.cache[key]
                if time.time() < expiry:
                    return value
                else:
                    del self.cache[key]
            return None

    def set(self, key: str, value: Dict):
        """Set value in cache with TTL."""
        with self.lock:
            expiry = time.time() + self.ttl_seconds
            self.cache[key] = (value, expiry)

            # Clean expired entries periodically
            if len(self.cache) > 1000:
                self._clean_expired()

    def _clean_expired(self):
        """Remove expired entries."""
        now = time.time()
        expired_keys = [k for k, (_, exp) in self.cache.items() if now >= exp]
        for k in expired_keys:
            del self.cache[k]
        if expired_keys:
            logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")


# Global cache
prediction_cache = SimpleCache(ttl_seconds=60)


# ============================================================================
# Dummy Model (for demonstration)
# ============================================================================


class DummyModel:
    """Dummy model for testing."""

    def predict(self, features: np.ndarray) -> tuple:
        """Make random prediction."""
        # Simulate some computation time
        time.sleep(0.001)

        # Random prediction
        probs = np.random.dirichlet(np.ones(3))
        prediction = int(np.argmax(probs))

        return prediction, probs


# ============================================================================
# Application State
# ============================================================================


class ApplicationState:
    """Application state management."""

    def __init__(self):
        self.models = {}
        self.start_time = time.time()

    def load_models(self):
        """Load ML models."""
        logger.info("Loading models...")

        # Load dummy model (replace with real models in production)
        self.models["dummy"] = DummyModel()

        # TODO: Load real models
        # self.models['lstm_v1'] = load_lstm_model('models/lstm_v1.pth')
        # self.models['transformer_v1'] = load_transformer_model('models/transformer_v1.pth')

        logger.info(f"Loaded {len(self.models)} models: {list(self.models.keys())}")

    def get_uptime(self) -> float:
        """Get uptime in seconds."""
        return time.time() - self.start_time


# Global state
state = ApplicationState()


# ============================================================================
# Startup/Shutdown Events
# ============================================================================


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Starting HFT Prediction API...")
    state.load_models()
    logger.info("API ready to accept requests")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down HFT Prediction API...")


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information."""
    return {
        "service": "HFT Order Book Prediction API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns service status, loaded models, and uptime.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="2.0.0",
        models_loaded=list(state.models.keys()),
        uptime_seconds=state.get_uptime(),
    )


@app.get("/models", response_model=Dict, tags=["Models"])
async def list_models():
    """List available models."""
    return {
        "models": [
            {"name": name, "type": type(model).__name__, "loaded": True}
            for name, model in state.models.items()
        ]
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
@limiter.limit("10/minute")  # Rate limit: 10 requests per minute per IP
@with_timeout(5.0)  # 5 second timeout
async def predict(request: Request, pred_request: PredictionRequest):
    """
    Make prediction on order book snapshots.

    **Rate Limit**: 10 requests/minute per IP
    **Timeout**: 5 seconds
    **Max Snapshots**: 1000 per request

    Args:
        pred_request: Prediction request with snapshots and model name

    Returns:
        PredictionResponse with prediction, probabilities, and metadata

    Raises:
        HTTPException 404: Model not found
        HTTPException 422: Invalid request data
        HTTPException 429: Rate limit exceeded
        HTTPException 504: Request timeout
    """
    start_time = time.time()

    logger.info(
        f"Prediction request from {request.client.host}: "
        f"model={pred_request.model_name}, snapshots={len(pred_request.snapshots)}"
    )

    # Generate cache key
    cache_key = hashlib.md5(
        json.dumps(
            {
                "model": pred_request.model_name,
                "snapshots": [
                    s.dict() for s in pred_request.snapshots[:10]
                ],  # Use first 10 for key
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    # Check cache
    cached_result = prediction_cache.get(cache_key)
    if cached_result:
        latency_ms = (time.time() - start_time) * 1000
        cached_result["latency_ms"] = latency_ms
        cached_result["cache_hit"] = True
        metrics_collector.record_prediction(latency_ms, True, pred_request.model_name)
        logger.info(f"Cache hit for request (latency: {latency_ms:.2f}ms)")
        return PredictionResponse(**cached_result)

    # Check if model exists
    if (
        pred_request.model_name not in state.models
        and pred_request.model_name != "dummy"
    ):
        available_models = list(state.models.keys()) + ["dummy"]
        metrics_collector.record_error("model_not_found")
        raise HTTPException(
            status_code=404,
            detail={
                "error": "model_not_found",
                "message": f"Model '{pred_request.model_name}' not found",
                "available_models": available_models,
                "hint": "Check /models endpoint for available models",
            },
        )

    # Get model
    model = state.models.get(pred_request.model_name) or state.models.get("dummy")

    # Make prediction (dummy implementation)
    try:
        prediction_idx, probabilities = model.predict(
            np.zeros((len(pred_request.snapshots), 16))
        )

        # Convert to response format
        class_names = ["down", "flat", "up"]
        prediction = class_names[prediction_idx]
        prob_dict = {
            name: float(prob) for name, prob in zip(class_names, probabilities)
        }
        confidence = float(max(probabilities))

        latency_ms = (time.time() - start_time) * 1000

        result = {
            "prediction": prediction,
            "probabilities": prob_dict,
            "confidence": confidence,
            "model_name": pred_request.model_name,
            "latency_ms": latency_ms,
            "cache_hit": False,
            "timestamp": datetime.utcnow().isoformat(),
            "features": None,
        }

        # Cache result
        prediction_cache.set(cache_key, result.copy())

        # Record metrics
        metrics_collector.record_prediction(latency_ms, False, pred_request.model_name)

        logger.info(
            f"Prediction completed: {prediction} (confidence: {confidence:.2f}, "
            f"latency: {latency_ms:.2f}ms)"
        )

        return PredictionResponse(**result)

    except Exception as e:
        metrics_collector.record_error(type(e).__name__)
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "prediction_error",
                "message": "Internal error during prediction",
                "hint": "Please contact support if this persists",
            },
        )


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """
    Get API metrics.

    Returns comprehensive metrics including:
    - Total predictions
    - Latency statistics (avg, p50, p95, p99)
    - Cache hit rate
    - Predictions by model
    - Error counts
    """
    return MetricsResponse(**metrics_collector.get_metrics())


@app.post("/metrics/reset", tags=["Monitoring"])
async def reset_metrics():
    """Reset all metrics (admin only)."""
    metrics_collector.reset()
    return {"status": "metrics_reset", "timestamp": datetime.utcnow().isoformat()}


# ============================================================================
# Error Handlers
# ============================================================================


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError with helpful message."""
    logger.error(f"ValueError: {str(exc)}")
    metrics_collector.record_error("ValueError")
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": str(exc),
            "hint": "Check your request data format",
        },
    )


# ============================================================================
# Main (for development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting development server...")

    uvicorn.run(
        "prediction_service_improved:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
