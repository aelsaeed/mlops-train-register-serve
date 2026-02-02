from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from mlops_trs.mlflow_utils import TrackingConfig, configure_tracking


@dataclass(frozen=True)
class TrainConfig:
    model_type: str
    random_seed: int
    test_size: float
    n_estimators: int
    max_depth: int | None
    c_value: float
    experiment_name: str
    tracking_uri: str | None
    artifact_location: str | None


def _load_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    dataset = load_iris(as_frame=True)
    features = dataset.data
    target = dataset.target
    return features, target


def _build_model(config: TrainConfig) -> Any:
    if config.model_type == "logreg":
        return LogisticRegression(
            C=config.c_value,
            max_iter=200,
            random_state=config.random_seed,
            solver="liblinear",
        )
    if config.model_type == "rf":
        return RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            random_state=config.random_seed,
        )
    message = f"Unsupported model_type: {config.model_type}"
    raise ValueError(message)


def train_model(config: TrainConfig) -> Dict[str, Any]:
    configure_tracking(
        TrackingConfig(
            tracking_uri=config.tracking_uri,
            experiment_name=config.experiment_name,
            artifact_location=config.artifact_location,
        )
    )

    features, target = _load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=target,
    )

    model = _build_model(config)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "model_type": config.model_type,
                "random_seed": config.random_seed,
                "test_size": config.test_size,
                "n_estimators": config.n_estimators,
                "max_depth": config.max_depth,
                "c_value": config.c_value,
            }
        )

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(model, artifact_path="model")
        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics_path = Path(tmp_dir) / "metrics.json"
            pd.DataFrame(
                {
                    "accuracy": [accuracy],
                    "test_rows": [len(x_test)],
                }
            ).to_json(metrics_path, orient="records")
            mlflow.log_artifact(str(metrics_path))

        run_id = run.info.run_id

    return {
        "run_id": run_id,
        "accuracy": float(accuracy),
        "model_uri": f"runs:/{run_id}/model",
        "feature_names": list(features.columns),
        "classes": sorted(np.unique(target).tolist()),
    }
