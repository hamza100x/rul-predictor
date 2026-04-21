"""Feature engineering helpers for the RUL project."""

from __future__ import annotations

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load a tabular dataset from disk."""
    return pd.read_csv(path)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a feature-ready copy of the input frame."""
    return frame.copy()
