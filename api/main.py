"""FastAPI application for RUL prediction."""

from __future__ import annotations

from pathlib import Path
import sys

from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import predict_rul

app = FastAPI(title="RUL Predictor API", version="0.1.0")


@app.get("/")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/predict")
def predict(model_path: str, data_path: str) -> dict[str, list[float]]:
    predictions = predict_rul(model_path=model_path, data_path=data_path)
    return {"predictions": predictions.tolist()}
