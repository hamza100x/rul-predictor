"""Inference helpers for CMAPSS RUL modeling."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def load_rf_model(model_path: str) -> object:
    """Load pickled RandomForest model."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_with_rf(
    model_path: str,
    data_path: str,
    feature_cols_path: str | None = None,
) -> pd.DataFrame:
    """Make RUL predictions using trained Random Forest baseline."""
    model = load_rf_model(model_path)
    frame = pd.read_parquet(data_path)

    if feature_cols_path:
        feature_cols = Path(feature_cols_path).read_text(encoding="utf-8").strip().split("\n")
    else:
        feature_cols = [c for c in frame.columns if c not in {"engine_id", "cycle", "rul"}]

    x = frame[feature_cols]
    pred = model.predict(x)
    return pd.DataFrame(
        {
            "engine_id": frame.get("engine_id", np.arange(len(frame))),
            "cycle": frame.get("cycle", np.arange(len(frame))),
            "predicted_rul": pred,
        }
    )


def predict_rul(model_path: str, data_path: str, feature_cols_path: str | None = None) -> pd.DataFrame:
    """Unified prediction interface supporting RF baseline."""
    return predict_with_rf(model_path, data_path, feature_cols_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict RUL values using trained models")
    parser.add_argument("--model", required=True, help="Path to trained model (RF pickle)")
    parser.add_argument("--data", required=True, help="Path to test data parquet")
    parser.add_argument("--features", default=None, help="Path to feature columns list")
    parser.add_argument("--output", default=None, help="Output CSV path (optional)")
    args = parser.parse_args()

    predictions = predict_rul(args.model, args.data, args.features)
    if args.output:
        predictions.to_csv(args.output, index=False)
        print(f"Predictions saved to: {args.output}")
    else:
        print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()
