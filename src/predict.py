"""Inference helpers for the RUL project."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def predict_rul(model_path: str, data_path: str) -> pd.Series:
    """Placeholder prediction routine."""
    _ = Path(model_path)
    frame = pd.read_csv(data_path)
    return pd.Series([0] * len(frame), name="predicted_rul")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict RUL values")
    parser.add_argument("--model", required=True, help="Path to a trained model")
    parser.add_argument("--data", required=True, help="Path to inference data")
    args = parser.parse_args()
    predictions = predict_rul(args.model, args.data)
    print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()
