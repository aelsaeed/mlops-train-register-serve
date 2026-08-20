from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import mlflow
import pandas as pd
import pytest
import sklearn
from fastapi.testclient import TestClient
from mlflow.models import Model
from mlflow.tracking import MlflowClient

import mlops_trs.registry as registry_module
from mlops_trs.api import create_app
from mlops_trs.constants import CANDIDATE_ALIAS, CHAMPION_ALIAS, IRIS_FEATURE_NAMES
from mlops_trs.registry import (
    PromotionGateError,
    RegistryConfig,
    promote_model,
    register_best_run,
)
from mlops_trs.training import TrainConfig, train_model


def test_promotion_rejects_a_version_that_is_not_the_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def get_model_version(self, *, name: str, version: str) -> SimpleNamespace:
            return SimpleNamespace(name=name, version=version, run_id="run-1")

        def get_model_version_by_alias(self, name: str, alias: str) -> SimpleNamespace:
            assert name == "iris-test"
            assert alias == CANDIDATE_ALIAS
            return SimpleNamespace(version="2")

    monkeypatch.setattr(registry_module, "MlflowClient", FakeClient)

    with pytest.raises(PromotionGateError, match="version 1 is not @candidate"):
        promote_model(
            model_name="iris-test",
            version="1",
            tracking_uri=None,
            minimum_accuracy=0.8,
        )


def test_promotion_rejects_non_finite_accuracy(monkeypatch: pytest.MonkeyPatch) -> None:
    tags: dict[str, str] = {}

    class FakeClient:
        def get_model_version(self, *, name: str, version: str) -> SimpleNamespace:
            return SimpleNamespace(name=name, version=version, run_id="run-1")

        def get_model_version_by_alias(self, name: str, alias: str) -> SimpleNamespace:
            assert name == "iris-test"
            assert alias == CANDIDATE_ALIAS
            return SimpleNamespace(version="1")

        def get_run(self, run_id: str) -> SimpleNamespace:
            assert run_id == "run-1"
            return SimpleNamespace(data=SimpleNamespace(metrics={"accuracy": float("nan")}))

        def set_model_version_tag(
            self,
            name: str,
            version: str,
            key: str,
            value: str,
        ) -> None:
            assert name == "iris-test"
            assert version == "1"
            tags[key] = value

    monkeypatch.setattr(registry_module, "MlflowClient", FakeClient)

    with pytest.raises(PromotionGateError, match="must be a finite value between 0 and 1"):
        promote_model(
            model_name="iris-test",
            version="1",
            tracking_uri=None,
            minimum_accuracy=0.8,
        )

    assert tags == {
        "promotion_gate": "failed",
        "promotion_gate_reason": "invalid_accuracy",
    }


def test_real_train_register_gate_promote_and_serve_lifecycle(tmp_path: Path) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_location = (tmp_path / "mlflow-artifacts").as_uri()
    output_dir = tmp_path / "exported-models"
    experiment_name = "lifecycle-test"
    model_name = "iris-lifecycle-test"

    result = train_model(
        TrainConfig(
            model_type="rf",
            random_seed=0,
            test_size=0.2,
            n_estimators=1,
            max_depth=1,
            c_value=1.0,
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            artifact_location=artifact_location,
            dataset_path=None,
            output_dir=str(output_dir),
        )
    )

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(result["run_id"])
    expected_metrics = {"accuracy", "precision_macro", "recall_macro", "f1_macro"}
    assert expected_metrics <= run.data.metrics.keys()
    assert all(0 <= run.data.metrics[name] <= 1 for name in expected_metrics)
    assert run.data.tags["dataset_sha256"] == result["dataset"]["sha256"]
    assert run.data.tags["source_sha256"] == result["source"]["sha256"]
    assert run.data.tags["model_version"] == result["model_version"]
    assert result["model_version"].startswith("v2-")
    assert len(result["dataset"]["sha256"]) == 64
    assert len(result["source"]["sha256"]) == 64
    assert result["source"]["runtime"]["scikit_learn"] == sklearn.__version__
    assert run.inputs.dataset_inputs
    assert {item.path for item in client.list_artifacts(result["run_id"])} >= {
        "dataset",
        "evaluation",
    }

    local_model_path = Path(result["local_model_path"])
    assert local_model_path.is_relative_to(tmp_path)
    assert local_model_path.is_dir()
    model_metadata = Model.load(local_model_path)
    assert model_metadata.signature is not None
    assert model_metadata.signature.inputs is not None
    assert model_metadata.signature.inputs.input_names() == list(IRIS_FEATURE_NAMES)
    assert model_metadata.signature.outputs is not None
    assert model_metadata.saved_input_example_info is not None
    input_example = cast(
        pd.DataFrame,
        model_metadata.load_input_example(str(local_model_path)),
    )
    assert list(input_example.columns) == list(IRIS_FEATURE_NAMES)

    registered = register_best_run(
        RegistryConfig(
            model_name=model_name,
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            artifact_location=artifact_location,
            run_id=result["run_id"],
        )
    )
    candidate = client.get_model_version_by_alias(model_name, CANDIDATE_ALIAS)
    assert candidate.version == registered.version
    assert candidate.run_id == result["run_id"]
    assert candidate.tags["deployment_status"] == CANDIDATE_ALIAS

    with pytest.raises(PromotionGateError, match="accuracy .* is below 0.8000"):
        promote_model(
            model_name=model_name,
            version=str(registered.version),
            tracking_uri=tracking_uri,
            minimum_accuracy=0.8,
        )
    failed_candidate = client.get_model_version(model_name, registered.version)
    assert failed_candidate.tags["promotion_gate"] == "failed"
    assert failed_candidate.tags["promotion_gate_reason"] == "accuracy_below_threshold"

    promoted = promote_model(
        model_name=model_name,
        version=str(registered.version),
        tracking_uri=tracking_uri,
        minimum_accuracy=0.5,
    )
    champion = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    assert promoted.model_version.version == registered.version
    assert promoted.alias == CHAMPION_ALIAS
    assert promoted.minimum_accuracy == 0.5
    assert champion.version == registered.version
    assert champion.tags["promotion_gate"] == "passed"

    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_name}@{CHAMPION_ALIAS}"
    api = TestClient(create_app(model_uri))
    assert api.get("/health/ready").status_code == 200
    response = api.post(
        "/predict",
        json={
            "instances": [
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                }
            ]
        },
    )
    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert response.json()["predictions"][0]["class_name"] in {
        "setosa",
        "versicolor",
        "virginica",
    }
