from pathlib import Path
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import api.main as api_main


client = TestClient(api_main.app)


def _mock_rf_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model_path = tmp_path / "rf_baseline.pkl"
    features_path = tmp_path / "rf_feature_columns.txt"

    model_path.write_bytes(b"placeholder")
    features_path.write_text("f1\nf2\n", encoding="utf-8")

    monkeypatch.setattr(
        api_main,
        "AVAILABLE_MODELS",
        {"rf": {"model": model_path, "features": features_path}},
    )


def test_root_endpoint() -> None:
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert "RUL Predictor API" in payload["message"]
    assert "v0.2" in payload["message"]


def test_health_endpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _mock_rf_model(monkeypatch, tmp_path)

    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in ("ok", "degraded")
    assert isinstance(payload["models_available"], dict)


def test_models_endpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _mock_rf_model(monkeypatch, tmp_path)

    response = client.get("/models")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, dict)
    assert "rf" in payload


def test_metrics_endpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "rf_metrics.json").write_text(
        '{"rmse": 15.5, "mae": 12.3}',
        encoding="utf-8",
    )

    monkeypatch.setattr(api_main, "MODELS_DIR", tmp_path)

    response = client.get("/metrics?model_name=rf")
    assert response.status_code == 200
    assert response.json()["rmse"] == 15.5


def test_metrics_endpoint_invalid_model() -> None:
    response = client.get("/metrics?model_name=invalid_model")
    assert response.status_code == 400
    assert "Unknown model" in response.json()["detail"]


def test_predict_endpoint_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_path = tmp_path / "test_fd001_features.parquet"
    data_path.write_bytes(b"placeholder")
    _mock_rf_model(monkeypatch, tmp_path)

    def fake_predict_rul(
        model_path: str,
        data_path: str,
        feature_cols_path: str | None = None,
    ) -> pd.DataFrame:
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
        json={"data_path": str(data_path), "model_name": "rf"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["num_samples"] == 2
    assert payload["model_type"] == "rf"
    assert len(payload["predictions"]) == 2


def test_predict_endpoint_missing_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _mock_rf_model(monkeypatch, tmp_path)

    response = client.post(
        "/predict",
        json={
            "data_path": str(tmp_path / "missing.parquet"),
            "model_name": "rf",
        },
    )

    assert response.status_code == 404
    assert "Data not found" in response.json()["detail"]


def test_predict_endpoint_invalid_model() -> None:
    response = client.post(
        "/predict",
        json={
            "data_path": "data/test.parquet",
            "model_name": "invalid_model",
        },
    )

    assert response.status_code == 422
