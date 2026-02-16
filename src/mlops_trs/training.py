from __future__ import annotations

import hashlib
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    dataset_path: str | None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _load_dataset(dataset_path: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    if dataset_path:
        frame = pd.read_csv(dataset_path)
        target = frame["target"]
        features = frame.drop(columns=["target"])
        return features, target

    dataset = load_iris(as_frame=True)
    return dataset.data, dataset.target


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


def _build_model_version(config: TrainConfig, features: pd.DataFrame, target: pd.Series) -> str:
    version_source = "|".join(
        [
            str(config.random_seed),
            str(config.model_type),
            str(config.n_estimators),
            str(config.max_depth),
            str(config.c_value),
            str(config.dataset_path or "iris"),
            str(features.shape),
            str(target.value_counts().to_dict()),
        ]
    )
    version_hash = hashlib.sha1(version_source.encode("utf-8")).hexdigest()[:8]
    return f"v1-{version_hash}"


def train_model(config: TrainConfig) -> dict[str, Any]:
    _seed_everything(config.random_seed)
    configure_tracking(
        TrackingConfig(
            tracking_uri=config.tracking_uri,
            experiment_name=config.experiment_name,
            artifact_location=config.artifact_location,
        )
    )

    features, target = _load_dataset(config.dataset_path)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=target,
    )

    model = _build_model(config)
    model_version = _build_model_version(config, features, target)

    with mlflow.start_run() as run:
        mlflow.set_tags({"lifecycle": "dev", "model_version": model_version})
        mlflow.log_params(
            {
                "model_type": config.model_type,
                "random_seed": config.random_seed,
                "test_size": config.test_size,
                "n_estimators": config.n_estimators,
                "max_depth": config.max_depth,
                "c_value": config.c_value,
                "dataset_path": config.dataset_path or "sklearn_iris",
            }
        )

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(model, artifact_path="model")
        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics_path = Path(tmp_dir) / "metrics.json"
            pd.DataFrame({"accuracy": [accuracy], "test_rows": [len(x_test)]}).to_json(
                metrics_path, orient="records"
            )
            mlflow.log_artifact(str(metrics_path))

        run_id = run.info.run_id

    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    local_model_dir = artifact_dir / f"model_{model_version}"
    if local_model_dir.exists():
        shutil.rmtree(local_model_dir)
    mlflow.sklearn.save_model(sk_model=model, path=str(local_model_dir))

    return {
        "run_id": run_id,
        "accuracy": float(accuracy),
        "model_uri": f"runs:/{run_id}/model",
        "feature_names": list(features.columns),
        "classes": sorted(np.unique(target).tolist()),
        "model_version": model_version,
        "local_model_path": str(local_model_dir),
    }
