"""
FastAPI Prediction Service for Order Book Imbalance Forecasting

Real-time API for serving model predictions with:
- Low-latency inference (<50ms)
- Redis caching
- Batch prediction support
- Health monitoring
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import torch
import numpy as np
from datetime import datetime
import redis
import json
import os
from loguru import logger

# Import models (in production, these would be properly loaded)
# from models.lstm_model import OrderBookLSTM


# Pydantic models for request/response
class OrderBookLevel(BaseModel):
    """Single order book level."""

    price: float
    volume: float


class OrderBookSnapshot(BaseModel):
    """Order book snapshot for prediction."""

    timestamp: int
    exchange: str
    symbol: str
    bids: List[List[float]] = Field(..., description="List of [price, volume] for bids")
    asks: List[List[float]] = Field(..., description="List of [price, volume] for asks")


class PredictionRequest(BaseModel):
    """Request for model prediction."""

    snapshots: List[OrderBookSnapshot] = Field(
        ..., description="Sequence of order book snapshots"
    )
    model_name: str = Field(
        default="lstm_v1", description="Model to use for prediction"
    )


class PredictionResponse(BaseModel):
    """Response containing prediction."""

    prediction: str = Field(..., description="Predicted class: up, down, or flat")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    confidence: float = Field(..., description="Confidence score (max probability)")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_name: str
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    models_loaded: List[str]
    redis_connected: bool


# Initialize FastAPI app
app = FastAPI(
    title="HFT Order Book Prediction API",
    description="Real-time order book imbalance forecasting",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global state
class AppState:
    """Application state container."""

    def __init__(self):
        self.models: Dict[str, torch.nn.Module] = {}
        self.redis_client: Optional[redis.Redis] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


state = AppState()


@app.on_event("startup")
async def startup_event():
    """Initialize models and connections on startup."""
    logger.info("Starting HFT Prediction API...")

    # Initialize Redis
    try:
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        state.redis_client = redis.Redis(
            host=redis_host, port=redis_port, decode_responses=True
        )
        state.redis_client.ping()
        logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        state.redis_client = None

    # Load models (placeholder - in production, load actual model weights)
    # model_path = "models/saved/lstm_v1.pth"
    # if os.path.exists(model_path):
    #     model = OrderBookLSTM(input_size=16, hidden_size=256, num_layers=2)
    #     model.load_state_dict(torch.load(model_path))
    #     model.to(state.device)
    #     model.eval()
    #     state.models["lstm_v1"] = model
    #     logger.info(f"Loaded model: lstm_v1")

    logger.info(f"API startup complete. Device: {state.device}")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "HFT Order Book Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    redis_ok = False
    if state.redis_client:
        try:
            state.redis_client.ping()
            redis_ok = True
        except:
            pass

    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        models_loaded=list(state.models.keys()),
        redis_connected=redis_ok,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Make prediction on order book sequence.

    This endpoint processes a sequence of order book snapshots and
    returns a prediction for future price direction.
    """
    start_time = datetime.utcnow()

    # Validate model exists
    if request.model_name not in state.models and request.model_name != "dummy":
        # For demo purposes, allow "dummy" model
        if request.model_name != "dummy":
            raise HTTPException(
                status_code=404, detail=f"Model '{request.model_name}' not found"
            )

    # Check cache (if Redis available)
    cache_key = None
    if state.redis_client:
        cache_key = f"pred:{request.model_name}:{hash(str(request.snapshots))}"
        try:
            cached = state.redis_client.get(cache_key)
            if cached:
                logger.info("Returning cached prediction")
                return PredictionResponse(**json.loads(cached))
        except Exception as e:
            logger.warning(f"Cache read error: {e}")

    # Extract features from snapshots (simplified for demo)
    # In production, this would use the actual feature engineering pipeline
    features = extract_features(request.snapshots)

    # Make prediction (dummy for now)
    if request.model_name == "dummy":
        # Random prediction for demo
        probs = np.random.dirichlet([1, 1, 1])
        pred_class = int(np.argmax(probs))
        class_names = ["down", "flat", "up"]
    else:
        # Real model prediction
        model = state.models[request.model_name]
        with torch.no_grad():
            input_tensor = torch.FloatTensor(features).unsqueeze(0).to(state.device)
            probs = model.predict_proba(input_tensor).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))
            class_names = ["down", "flat", "up"]

    # Calculate latency
    latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

    # Build response
    response = PredictionResponse(
        prediction=class_names[pred_class],
        probabilities={
            "down": float(probs[0]),
            "flat": float(probs[1]),
            "up": float(probs[2]),
        },
        confidence=float(np.max(probs)),
        timestamp=datetime.utcnow().isoformat(),
        model_name=request.model_name,
        latency_ms=latency_ms,
    )

    # Cache result
    if state.redis_client and cache_key:
        try:
            state.redis_client.setex(
                cache_key, 60, json.dumps(response.dict())  # 60 second TTL
            )
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    logger.info(
        f"Prediction: {class_names[pred_class]} (confidence: {response.confidence:.3f}, latency: {latency_ms:.2f}ms)"
    )

    return response


def extract_features(snapshots: List[OrderBookSnapshot]) -> np.ndarray:
    """
    Extract features from order book snapshots.

    This is a simplified version for demo purposes.
    In production, this would use the full feature engineering pipeline.
    """
    features_list = []

    for snapshot in snapshots:
        # Extract top-of-book features
        if snapshot.bids and snapshot.asks:
            bid_price, bid_vol = snapshot.bids[0]
            ask_price, ask_vol = snapshot.asks[0]

            mid_price = (bid_price + ask_price) / 2
            spread = ask_price - bid_price
            spread_bps = (spread / mid_price * 10000) if mid_price > 0 else 0

            # Simple features
            features = [
                mid_price,
                spread,
                spread_bps,
                bid_vol,
                ask_vol,
                bid_vol - ask_vol,  # volume imbalance
            ]

            # Pad to fixed size (16 features for demo)
            while len(features) < 16:
                features.append(0.0)

            features_list.append(features[:16])

    # Convert to sequence format (batch=1, sequence_length, features)
    return np.array(features_list[-100:])  # Last 100 snapshots


@app.get("/models", tags=["Models"])
async def list_models():
    """List available models."""
    return {"models": list(state.models.keys()), "device": str(state.device)}


@app.get("/metrics", tags=["Metrics"])
async def get_metrics():
    """Get API metrics."""
    # In production, this would return actual metrics from monitoring
    return {
        "total_predictions": 0,
        "avg_latency_ms": 0,
        "cache_hit_rate": 0,
        "timestamp": datetime.utcnow().isoformat(),
    }


# Run with: uvicorn src.api.prediction_service:app --reload
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
