"""FastAPI application for RUL prediction."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import predict_rul

app = FastAPI(
    title="RUL Predictor API",
    description="Predict Remaining Useful Life (RUL) for turbofan engines",
    version="0.1.0",
)

MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"

DEFAULT_RF_MODEL = MODELS_DIR / "rf_baseline.pkl"
DEFAULT_FEATURE_COLS = MODELS_DIR / "rf_feature_columns.txt"


def _resolve_path(path_value: str) -> Path:
    """Resolve user-provided path relative to project root when needed."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


class PredictionRequest(BaseModel):
    """Request body for batch predictions."""

    data_path: str
    model_path: str | None = None
    feature_cols_path: str | None = None


class PredictionResponse(BaseModel):
    """Response body for predictions."""

    predictions: list[dict[str, Any]]
    num_samples: int
    model_used: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    models_available: list[str]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Check API health and available models."""
    available = []
    if DEFAULT_RF_MODEL.exists():
        available.append("rf_baseline")
    return HealthResponse(status="ok", models_available=available)


@app.get("/metrics")
def get_metrics() -> dict[str, Any]:
    """Get baseline model metrics."""
    metrics_path = MODELS_DIR / "rf_metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    return {"error": "Metrics not found"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Make RUL predictions on test data."""
    model_path = _resolve_path(request.model_path or str(DEFAULT_RF_MODEL))
    data_path = _resolve_path(request.data_path)
    feature_cols_path = _resolve_path(request.feature_cols_path or str(DEFAULT_FEATURE_COLS))

    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")
    if not data_path.exists():
        raise HTTPException(status_code=404, detail=f"Data not found: {data_path}")

    try:
        pred_df = predict_rul(str(model_path), str(data_path), str(feature_cols_path))
    except ImportError as exc:
        # Most commonly triggered when parquet engines are missing in the active environment.
        raise HTTPException(
            status_code=500,
            detail=(
                "Missing parquet dependency in the API runtime environment. "
                "Install 'pyarrow' (or 'fastparquet') in the same environment used to run uvicorn."
            ),
        ) from exc
    except KeyError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Missing expected feature column: {exc}",
        ) from exc
    predictions = pred_df.to_dict("records")

    return PredictionResponse(
        predictions=predictions,
        num_samples=len(predictions),
        model_used="rf_baseline",
    )


@app.get("/")
def root() -> dict[str, str]:
    """API root."""
    return {
        "message": "RUL Predictor API",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }
