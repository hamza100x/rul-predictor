"""FastAPI application for RUL prediction."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import predict_rul

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RUL Predictor API",
    description="Predict Remaining Useful Life (RUL) for turbofan engines",
    version="0.2.0",
)

MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"

# Available models
DEFAULT_RF_MODEL = MODELS_DIR / "rf_baseline.pkl"
DEFAULT_LSTM_MODEL = MODELS_DIR / "lstm_model.pth"
DEFAULT_FEATURE_COLS = MODELS_DIR / "rf_feature_columns.txt"

AVAILABLE_MODELS = {
    "rf": {"model": DEFAULT_RF_MODEL, "features": DEFAULT_FEATURE_COLS},
    "lstm": {"model": DEFAULT_LSTM_MODEL, "features": None},
}


def _resolve_path(path_value: str) -> Path:
    """Resolve user-provided path relative to project root when needed."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


class PredictionRequest(BaseModel):
    """Request body for batch predictions."""

    data_path: str = Field(..., description="Path to parquet data file (relative or absolute)")
    model_name: Literal["rf", "lstm"] = Field("rf", description="Model to use for prediction")
    model_path: str | None = Field(None, description="Optional custom model path")
    feature_cols_path: str | None = Field(None, description="Optional custom feature columns path")

    class Config:
        json_schema_extra = {
            "example": {
                "data_path": "data/processed/fd001_test.parquet",
                "model_name": "rf",
            }
        }


class PredictionResponse(BaseModel):
    """Response body for predictions."""

    predictions: list[dict[str, Any]]
    num_samples: int
    model_used: str
    model_type: Literal["rf", "lstm"]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    models_available: dict[str, bool]
    version: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Check API health and available models."""
    models_status = {}
    for model_name, config in AVAILABLE_MODELS.items():
        exists = config["model"].exists()
        models_status[model_name] = exists
        logger.info(f"Model {model_name}: {'available' if exists else 'missing'}")
    
    return HealthResponse(
        status="ok" if any(models_status.values()) else "degraded",
        models_available=models_status,
        version="0.2.0",
    )


@app.get("/models")
def list_models() -> dict[str, Any]:
    """List all available models with their metadata."""
    models_info = {}
    for model_name, config in AVAILABLE_MODELS.items():
        model_exists = config["model"].exists()
        metrics_path = MODELS_DIR / f"{model_name}_metrics.json"
        metrics = {}
        
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse metrics for {model_name}")
        
        models_info[model_name] = {
            "available": model_exists,
            "path": str(config["model"]),
            "metrics": metrics,
        }
    
    return models_info


@app.get("/metrics")
def get_metrics(model_name: str = "rf") -> dict[str, Any]:
    """Get model metrics."""
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model_name}. Available: {list(AVAILABLE_MODELS.keys())}",
        )
    
    metrics_path = MODELS_DIR / f"{model_name}_metrics.json"
    if metrics_path.exists():
        try:
            return json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse metrics for {model_name}: {e}")
            raise HTTPException(status_code=500, detail="Failed to parse metrics") from e
    
    raise HTTPException(status_code=404, detail=f"Metrics not found for model: {model_name}")


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Make RUL predictions on test data."""
    logger.info(f"Prediction request: model={request.model_name}, data={request.data_path}")
    
    # Validate model
    if request.model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {request.model_name}. Available: {list(AVAILABLE_MODELS.keys())}",
        )
    
    model_config = AVAILABLE_MODELS[request.model_name]
    model_path = _resolve_path(request.model_path or str(model_config["model"]))
    data_path = _resolve_path(request.data_path)
    feature_cols_path = _resolve_path(
        request.feature_cols_path or str(model_config["features"])
    ) if model_config["features"] else None

    # Validate paths
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")
    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        raise HTTPException(status_code=404, detail=f"Data not found: {data_path}")

    try:
        logger.info(f"Running prediction with {request.model_name} model")
        pred_df = predict_rul(str(model_path), str(data_path), str(feature_cols_path) if feature_cols_path else None)
        predictions = pred_df.to_dict("records")
        
        logger.info(f"Successfully generated {len(predictions)} predictions")
        return PredictionResponse(
            predictions=predictions,
            num_samples=len(predictions),
            model_used=str(model_path),
            model_type=request.model_name,
        )
    except ImportError as exc:
        logger.error(f"Missing dependency: {exc}")
        raise HTTPException(
            status_code=500,
            detail=(
                "Missing parquet dependency in the API runtime environment. "
                "Install 'pyarrow' (or 'fastparquet') in the same environment used to run uvicorn."
            ),
        ) from exc
    except KeyError as exc:
        logger.error(f"Missing feature column: {exc}")
        raise HTTPException(
            status_code=400,
            detail=f"Missing expected feature column: {exc}",
        ) from exc
    except Exception as exc:
        logger.error(f"Prediction failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc


@app.get("/")
def root() -> dict[str, str]:
    """API root with available endpoints."""
    return {
        "message": "RUL Predictor API v0.2.0",
        "docs": "/docs",
        "health": "/health",
        "models": "/models",
        "metrics": "/metrics?model_name=rf",
        "predict": "/predict (POST)",
    }
