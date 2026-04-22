"""Feature engineering helpers for CMAPSS RUL modeling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

CMAPSS_COLUMNS = (
    ["engine_id", "cycle"]
    + [f"op_{i}" for i in range(1, 4)]
    + [f"s{i}" for i in range(1, 22)]
)


def load_data(path: str) -> pd.DataFrame:
    """Load a tabular dataset from disk."""
    return pd.read_csv(path)


def load_cmapss_subset(data_dir: str | Path, subset: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load CMAPSS train, test and RUL files for the given subset."""
    root = Path(data_dir)
    train = pd.read_csv(
        root / f"train_{subset}.txt",
        sep=r"\s+",
        header=None,
        engine="python",
    )
    test = pd.read_csv(
        root / f"test_{subset}.txt",
        sep=r"\s+",
        header=None,
        engine="python",
    )
    rul = pd.read_csv(
        root / f"RUL_{subset}.txt",
        sep=r"\s+",
        header=None,
        engine="python",
    )

    train.columns = CMAPSS_COLUMNS
    test.columns = CMAPSS_COLUMNS
    rul.columns = ["rul"]
    return train, test, rul


def add_train_rul_labels(train_df: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    """Compute per-row train RUL labels and optionally cap them."""
    frame = train_df.copy()
    max_cycle = frame.groupby("engine_id")["cycle"].transform("max")
    frame["rul"] = (max_cycle - frame["cycle"]).clip(lower=0, upper=cap)
    return frame


def add_test_rul_labels(test_df: pd.DataFrame, rul_df: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    """Attach true RUL labels to test rows using per-engine end-of-life offsets."""
    frame = test_df.copy()
    engine_max = frame.groupby("engine_id")["cycle"].transform("max")
    base_rul = rul_df["rul"].reindex(frame["engine_id"].values - 1).to_numpy()
    frame["rul"] = (engine_max - frame["cycle"] + base_rul).clip(lower=0, upper=cap)
    return frame


def select_feature_columns(frame: pd.DataFrame, drop_sensors: list[str] | None = None) -> list[str]:
    """Build a stable feature list from operational settings and selected sensors."""
    drop_sensors = drop_sensors or []
    sensor_cols = [c for c in frame.columns if c.startswith("s") and c not in drop_sensors]
    op_cols = ["op_1", "op_2", "op_3"]
    return op_cols + sensor_cols


def add_degradation_features(
    frame: pd.DataFrame,
    sensor_cols: list[str],
    rolling_windows: tuple[int, ...] = (5, 10),
) -> pd.DataFrame:
    """Add delta and rolling statistics for each sensor per engine."""
    out = frame.copy().sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    group = out.groupby("engine_id", sort=False)

    for sensor in sensor_cols:
        out[f"{sensor}_delta1"] = group[sensor].diff().fillna(0.0)

    for window in rolling_windows:
        for sensor in sensor_cols:
            rolling = group[sensor].rolling(window=window, min_periods=1)
            out[f"{sensor}_roll_mean_{window}"] = rolling.mean().reset_index(level=0, drop=True)
            out[f"{sensor}_roll_std_{window}"] = (
                rolling.std().reset_index(level=0, drop=True).fillna(0.0)
            )

    return out


def fit_transform_scaler(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit scaler on train features and transform train/test."""
    scaler = StandardScaler()
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    train_scaled[feature_cols] = scaler.fit_transform(train_scaled[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_scaled[feature_cols])
    return train_scaled, test_scaled, scaler


def make_lstm_sequences(
    frame: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int = 30,
    target_col: str = "rul",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create fixed-length per-engine sequences for sequence models."""
    frame = frame.sort_values(["engine_id", "cycle"])
    x_list: list[np.ndarray] = []
    y_list: list[float] = []
    engine_ids: list[int] = []

    for engine_id, group in frame.groupby("engine_id"):
        values = group[feature_cols].to_numpy(dtype=np.float32)
        targets = group[target_col].to_numpy(dtype=np.float32)
        if len(group) < seq_len:
            continue
        for end_idx in range(seq_len - 1, len(group)):
            start_idx = end_idx - seq_len + 1
            x_list.append(values[start_idx : end_idx + 1])
            y_list.append(targets[end_idx])
            engine_ids.append(int(engine_id))

    return np.array(x_list), np.array(y_list), np.array(engine_ids)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible placeholder used by existing scripts."""
    return frame.copy()
