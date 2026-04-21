"""Training entrypoint for the RUL project."""

from __future__ import annotations

import argparse
from pathlib import Path

from features import build_features, load_data


def train(data_path: str, output_dir: str) -> None:
    """Run the training pipeline placeholder."""
    frame = load_data(data_path)
    features = build_features(frame)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    features.to_parquet(Path(output_dir) / "train_features.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the RUL model")
    parser.add_argument("--data", required=True, help="Path to training data")
    parser.add_argument("--output-dir", default="models", help="Directory for artifacts")
    args = parser.parse_args()
    train(args.data, args.output_dir)


if __name__ == "__main__":
    main()
