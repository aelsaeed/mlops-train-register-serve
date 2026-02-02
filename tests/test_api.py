from __future__ import annotations

from pathlib import Path

import mlflow
from fastapi.testclient import TestClient

from app.main import create_app
from mlops_trs.training import TrainConfig, train_model


def test_predict_endpoint_contract(tmp_path: Path) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_location = (tmp_path / "artifacts").as_uri()
    result = train_model(
        TrainConfig(
            model_type="logreg",
            random_seed=123,
            test_size=0.2,
            n_estimators=10,
            max_depth=3,
            c_value=1.0,
            experiment_name="test-experiment",
            tracking_uri=tracking_uri,
            artifact_location=artifact_location,
        )
    )

    mlflow.set_tracking_uri(tracking_uri)
    app = create_app(result["model_uri"])
    client = TestClient(app)

    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert response.status_code == 200
    payload = response.json()
    assert "predictions" in payload
    assert isinstance(payload["predictions"], list)
    assert "model_uri" in payload
