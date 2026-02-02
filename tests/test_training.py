from __future__ import annotations

import json
from pathlib import Path

import mlflow
from mlops_trs.training import TrainConfig, train_model


def test_training_runs(tmp_path: Path) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_location = (tmp_path / "artifacts").as_uri()
    result = train_model(
        TrainConfig(
            model_type="logreg",
            random_seed=7,
            test_size=0.2,
            n_estimators=50,
            max_depth=4,
            c_value=1.0,
            experiment_name="test-experiment",
            tracking_uri=tracking_uri,
            artifact_location=artifact_location,
        )
    )

    mlflow.set_tracking_uri(tracking_uri)
    run = mlflow.get_run(result["run_id"])
    assert "accuracy" in run.data.metrics


def test_model_loads(tmp_path: Path) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_location = (tmp_path / "artifacts").as_uri()
    result = train_model(
        TrainConfig(
            model_type="rf",
            random_seed=11,
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
    model = mlflow.pyfunc.load_model(result["model_uri"])
    predictions = model.predict([[5.1, 3.5, 1.4, 0.2]])
    assert len(predictions) == 1


def test_train_output_json(tmp_path: Path) -> None:
    output_path = tmp_path / "train_output.json"
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_location = (tmp_path / "artifacts").as_uri()
    result = train_model(
        TrainConfig(
            model_type="logreg",
            random_seed=21,
            test_size=0.2,
            n_estimators=10,
            max_depth=3,
            c_value=1.0,
            experiment_name="test-experiment",
            tracking_uri=tracking_uri,
            artifact_location=artifact_location,
        )
    )
    output_path.write_text(json.dumps(result, indent=2))
    data = json.loads(output_path.read_text())
    assert data["run_id"] == result["run_id"]
