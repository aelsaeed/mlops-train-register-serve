from __future__ import annotations

from typing import NoReturn

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mlops_trs.api import create_app
from mlops_trs.constants import IRIS_FEATURE_NAMES

MODEL_URI = "models:/iris-classifier@champion"
SETOSA = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}
VIRGINICA = {
    "sepal_length": 6.5,
    "sepal_width": 3.0,
    "petal_length": 5.8,
    "petal_width": 2.2,
}


class FakeIrisModel:
    def __init__(self) -> None:
        self.frames: list[pd.DataFrame] = []

    def predict(self, frame: pd.DataFrame) -> list[int]:
        self.frames.append(frame.copy())
        return [0 if value < 2 else 2 for value in frame["petal_length"]]

    def predict_proba(self, frame: pd.DataFrame) -> list[list[float]]:
        return [
            [0.95, 0.04, 0.01] if value < 2 else [0.01, 0.04, 0.95]
            for value in frame["petal_length"]
        ]


def _client_with_model() -> tuple[TestClient, FakeIrisModel, list[str]]:
    model = FakeIrisModel()
    loaded_uris: list[str] = []

    def loader(model_uri: str) -> FakeIrisModel:
        loaded_uris.append(model_uri)
        return model

    return TestClient(create_app(MODEL_URI, model_loader=loader)), model, loaded_uris


def test_liveness_readiness_and_model_metadata_are_distinct() -> None:
    client, _, loaded_uris = _client_with_model()

    assert client.get("/health/live").json() == {"status": "ok"}
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/model").json() == {
        "model_uri": MODEL_URI,
        "loaded": False,
        "feature_names": list(IRIS_FEATURE_NAMES),
    }
    assert loaded_uris == []

    ready = client.get("/health/ready")

    assert ready.status_code == 200
    assert ready.json() == {"status": "ready"}
    assert loaded_uris == [MODEL_URI]
    assert client.get("/model").json()["loaded"] is True


def test_readiness_failure_is_sanitized_and_liveness_stays_healthy() -> None:
    def failing_loader(_: str) -> NoReturn:
        raise RuntimeError("registry credentials leaked here")

    client = TestClient(create_app(MODEL_URI, model_loader=failing_loader))

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "Model is unavailable"}
    assert "credentials" not in response.text
    assert client.get("/health/live").status_code == 200
    assert client.get("/model").json()["loaded"] is False


def test_batch_prediction_uses_named_features_and_returns_confidence() -> None:
    client, model, loaded_uris = _client_with_model()

    response = client.post("/predict", json={"instances": [SETOSA, VIRGINICA]})

    assert response.status_code == 200
    assert response.json() == {
        "predictions": [
            {"class_id": 0, "class_name": "setosa", "confidence": 0.95},
            {"class_id": 2, "class_name": "virginica", "confidence": 0.95},
        ],
        "model_uri": MODEL_URI,
        "count": 2,
    }
    assert loaded_uris == [MODEL_URI]
    assert len(model.frames) == 1
    assert list(model.frames[0].columns) == list(IRIS_FEATURE_NAMES)
    assert model.frames[0].to_dict(orient="records") == [SETOSA, VIRGINICA]


@pytest.mark.parametrize(
    "payload",
    [
        {"features": [5.1, 3.5, 1.4, 0.2]},
        {"instances": []},
        {"instances": [{key: value for key, value in SETOSA.items() if key != "petal_width"}]},
        {"instances": [{**SETOSA, "unknown": 1.0}]},
        {"instances": [{**SETOSA, "sepal_length": 0}]},
        {"instances": [SETOSA] * 101},
    ],
)
def test_invalid_prediction_payload_is_rejected_before_model_loading(
    payload: dict[str, object],
) -> None:
    client, _, loaded_uris = _client_with_model()

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert loaded_uris == []


def test_prediction_failure_returns_503_and_increments_error_metric() -> None:
    class BrokenModel:
        def predict(self, _: pd.DataFrame) -> NoReturn:
            raise RuntimeError("unsafe implementation detail")

    def loader(_: str) -> BrokenModel:
        return BrokenModel()

    client = TestClient(create_app(MODEL_URI, model_loader=loader))

    response = client.post("/predict", json={"instances": [SETOSA]})
    metrics = client.get("/metrics")

    assert response.status_code == 503
    assert response.json() == {"detail": "Model inference is unavailable"}
    assert "unsafe" not in response.text
    assert "mlops_prediction_requests_total 1.0" in metrics.text
    assert "mlops_prediction_errors_total 1.0" in metrics.text


def test_metrics_use_prometheus_content_type_and_track_batch_rows() -> None:
    client, _, _ = _client_with_model()
    assert client.post("/predict", json={"instances": [SETOSA, VIRGINICA]}).status_code == 200

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain;")
    assert "version=" in response.headers["content-type"]
    assert not response.text.startswith('"')
    assert "mlops_prediction_requests_total 1.0" in response.text
    assert "mlops_prediction_rows_total 2.0" in response.text
    assert "mlops_prediction_errors_total 0.0" in response.text
    assert "mlops_prediction_duration_seconds_count 1.0" in response.text
