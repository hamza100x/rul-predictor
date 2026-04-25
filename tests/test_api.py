from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

import api.main as api_main


client = TestClient(api_main.app)


def test_root_endpoint() -> None:
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["message"] == "RUL Predictor API"


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "models_available" in payload


def test_metrics_endpoint() -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, dict)
    assert "rmse" in payload or "error" in payload


def test_predict_endpoint_success(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "rf_baseline.pkl"
    features_path = tmp_path / "rf_feature_columns.txt"
    data_path = tmp_path / "test_fd001_features.parquet"

    model_path.write_bytes(b"placeholder")
    features_path.write_text("f1\nf2\n", encoding="utf-8")
    data_path.write_bytes(b"placeholder")

    monkeypatch.setattr(api_main, "DEFAULT_RF_MODEL", model_path)
    monkeypatch.setattr(api_main, "DEFAULT_FEATURE_COLS", features_path)

    def fake_predict_rul(model_path: str, data_path: str, feature_cols_path: str | None = None) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "engine_id": [1, 1],
                "cycle": [1, 2],
                "predicted_rul": [120.0, 119.0],
            }
        )

    monkeypatch.setattr(api_main, "predict_rul", fake_predict_rul)

    response = client.post(
        "/predict",
        json={
            "data_path": str(data_path),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["num_samples"] == 2
    assert payload["model_used"] == "rf_baseline"
    assert len(payload["predictions"]) == 2


def test_predict_endpoint_missing_data(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "rf_baseline.pkl"
    features_path = tmp_path / "rf_feature_columns.txt"

    model_path.write_bytes(b"placeholder")
    features_path.write_text("f1\nf2\n", encoding="utf-8")

    monkeypatch.setattr(api_main, "DEFAULT_RF_MODEL", model_path)
    monkeypatch.setattr(api_main, "DEFAULT_FEATURE_COLS", features_path)

    response = client.post(
        "/predict",
        json={
            "data_path": str(tmp_path / "missing.parquet"),
        },
    )

    assert response.status_code == 404
    assert "Data not found" in response.json()["detail"]
