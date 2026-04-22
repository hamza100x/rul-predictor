"""Training entrypoint for the RUL project."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the asymmetric NASA scoring metric used in CMAPSS."""
    diff = y_pred - y_true
    over = diff >= 0
    score = np.where(
        over,
        np.exp(diff / 10.0) - 1.0,
        np.exp(-diff / 13.0) - 1.0,
    )
    return float(np.sum(score))


def load_feature_table(path: str | Path) -> pd.DataFrame:
    """Load a parquet feature table and validate required columns."""
    frame = pd.read_parquet(path)
    required = {"engine_id", "cycle", "rul"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return frame


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Split tabular frame into model inputs and target."""
    feature_cols = [c for c in frame.columns if c not in {"engine_id", "cycle", "rul"}]
    x = frame[feature_cols]
    y = frame["rul"]
    return x, y, feature_cols


def train(
    train_path: str,
    test_path: str,
    output_dir: str,
    n_estimators: int = 400,
    max_depth: int | None = 18,
    random_state: int = 42,
) -> None:
    """Train and evaluate a baseline Random Forest for RUL prediction."""
    train_df = load_feature_table(train_path)
    test_df = load_feature_table(test_path)
    x_train, y_train, feature_cols = split_xy(train_df)
    x_test, y_test, _ = split_xy(test_df)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))
    score = nasa_score(y_test.to_numpy(), pred)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "rf_baseline.pkl", "wb") as f:
        pickle.dump(model, f)

    pd.DataFrame(
        {
            "engine_id": test_df["engine_id"],
            "cycle": test_df["cycle"],
            "y_true": y_test,
            "y_pred": pred,
        }
    ).to_csv(out_dir / "rf_test_predictions.csv", index=False)

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "nasa_score": score,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "num_features": len(feature_cols),
        "num_train_rows": int(len(train_df)),
        "num_test_rows": int(len(test_df)),
    }
    (out_dir / "rf_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "rf_feature_columns.txt").write_text("\n".join(feature_cols), encoding="utf-8")

    print("Random Forest baseline training complete")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"NASA score: {score:.2f}")
    print(f"Saved artifacts to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the RUL model")
    parser.add_argument(
        "--train-data",
        default="data/processed/train_fd001_features.parquet",
        help="Path to processed training features parquet",
    )
    parser.add_argument(
        "--test-data",
        default="data/processed/test_fd001_features.parquet",
        help="Path to processed test features parquet",
    )
    parser.add_argument("--output-dir", default="models", help="Directory for artifacts")
    parser.add_argument("--n-estimators", type=int, default=400, help="Number of trees")
    parser.add_argument("--max-depth", type=int, default=18, help="Maximum tree depth")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    train(
        train_path=args.train_data,
        test_path=args.test_data,
        output_dir=args.output_dir,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
