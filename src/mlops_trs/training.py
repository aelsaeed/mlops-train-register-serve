from __future__ import annotations

import hashlib
import json
import platform
import random
import shutil
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow.data import from_pandas  # type: ignore[attr-defined]
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, DataType, Schema
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops_trs.constants import IRIS_CLASS_NAMES, IRIS_FEATURE_NAMES
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
    output_dir: str = "artifacts"


class DatasetValidationError(ValueError):
    """Raised when training data violates the model's input contract."""


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _load_dataset(dataset_path: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    if dataset_path:
        path = Path(dataset_path)
        if not path.is_file():
            raise DatasetValidationError(f"Dataset not found: {path}")
        frame = pd.read_csv(path)
        expected_columns = {*IRIS_FEATURE_NAMES, "target"}
        missing = sorted(expected_columns - set(frame.columns))
        unexpected = sorted(set(frame.columns) - expected_columns)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing columns: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected columns: {', '.join(unexpected)}")
            raise DatasetValidationError("Invalid dataset schema (" + "; ".join(details) + ")")
        target = frame["target"]
        features = frame.loc[:, list(IRIS_FEATURE_NAMES)]
    else:
        dataset = load_iris(as_frame=True)
        features = dataset.data.copy()
        features.columns = list(IRIS_FEATURE_NAMES)
        target = dataset.target

    if features.empty:
        raise DatasetValidationError("Dataset must contain at least one row")
    if features.isna().any().any() or target.isna().any():
        raise DatasetValidationError("Dataset must not contain missing values")
    if not all(pd.api.types.is_numeric_dtype(features[column]) for column in features):
        raise DatasetValidationError("All feature columns must be numeric")
    if not pd.api.types.is_numeric_dtype(target):
        raise DatasetValidationError("Target column must be numeric")
    if (
        not np.isfinite(features.to_numpy(dtype=float)).all()
        or not np.isfinite(target.to_numpy(dtype=float)).all()
    ):
        raise DatasetValidationError("Dataset values must be finite")
    if ((features <= 0) | (features > 10)).any().any():
        raise DatasetValidationError("Feature values must be greater than 0 and at most 10")
    if target.nunique() < 2:
        raise DatasetValidationError("Target column must contain at least two classes")
    target_values = target.to_numpy(dtype=float)
    if not np.equal(target_values, np.floor(target_values)).all():
        raise DatasetValidationError("Target values must be integer class IDs")
    integer_target = target.astype(int)
    expected_classes = set(IRIS_CLASS_NAMES)
    actual_classes = set(integer_target.unique())
    if actual_classes != expected_classes:
        expected = ", ".join(str(value) for value in sorted(expected_classes))
        raise DatasetValidationError(f"Target column must contain Iris classes: {expected}")
    if int(target.value_counts().min()) < 2:
        raise DatasetValidationError("Every target class must contain at least two rows")
    return features.astype(float), integer_target


def _build_model(config: TrainConfig) -> Any:
    if config.model_type == "logreg":
        return Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=config.c_value,
                        max_iter=500,
                        random_state=config.random_seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
    if config.model_type == "rf":
        return RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            random_state=config.random_seed,
        )
    message = f"Unsupported model_type: {config.model_type}"
    raise ValueError(message)


def _dataset_sha256(features: pd.DataFrame, target: pd.Series) -> str:
    frame = features.copy()
    frame["target"] = target.to_numpy()
    canonical_csv = frame.to_csv(index=False, lineterminator="\n", float_format="%.12g")
    return hashlib.sha256(canonical_csv.encode("utf-8")).hexdigest()


def _source_sha256() -> str:
    digest = hashlib.sha256()
    source_root = Path(__file__).resolve().parent
    for source_path in sorted(source_root.glob("*.py")):
        digest.update(source_path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "mlflow": mlflow.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


def _build_model_version(
    config: TrainConfig,
    dataset_sha256: str,
    *,
    source_sha256: str | None = None,
    runtime_versions: dict[str, str] | None = None,
) -> str:
    version_source = json.dumps(
        {
            "config": {
                "c_value": config.c_value,
                "max_depth": config.max_depth,
                "model_type": config.model_type,
                "n_estimators": config.n_estimators,
                "random_seed": config.random_seed,
                "test_size": config.test_size,
            },
            "dataset_sha256": dataset_sha256,
            "runtime": runtime_versions or _runtime_versions(),
            "source_sha256": source_sha256 or _source_sha256(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    version_hash = hashlib.sha256(version_source.encode("utf-8")).hexdigest()[:12]
    return f"v2-{version_hash}"


def _validate_split(config: TrainConfig, target: pd.Series) -> None:
    if not 0 < config.test_size < 1:
        raise DatasetValidationError("test_size must be between 0 and 1")
    class_count = int(target.nunique())
    test_rows = int(np.ceil(len(target) * config.test_size))
    train_rows = len(target) - test_rows
    if min(test_rows, train_rows) < class_count:
        raise DatasetValidationError(
            "Train and test splits must each contain at least one row per target class"
        )


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
    _validate_split(config, target)
    dataset_sha256 = _dataset_sha256(features, target)
    source_sha256 = _source_sha256()
    runtime_versions = _runtime_versions()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=target,
    )

    model = _build_model(config)
    model_version = _build_model_version(
        config,
        dataset_sha256,
        source_sha256=source_sha256,
        runtime_versions=runtime_versions,
    )
    started_at = time.perf_counter()

    with mlflow.start_run() as run:
        mlflow.set_tags(
            {
                "lifecycle": "training",
                "model_version": model_version,
                "dataset_sha256": dataset_sha256,
                "source_sha256": source_sha256,
            }
        )
        mlflow.log_params(
            {
                "model_type": config.model_type,
                "random_seed": config.random_seed,
                "test_size": config.test_size,
                "n_estimators": config.n_estimators,
                "max_depth": config.max_depth,
                "c_value": config.c_value,
                "dataset_path": config.dataset_path or "sklearn_iris",
                **{f"runtime_{name}": version for name, version in runtime_versions.items()},
            }
        )

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_macro": float(
                precision_score(y_test, predictions, average="macro", zero_division=0)
            ),
            "recall_macro": float(
                recall_score(y_test, predictions, average="macro", zero_division=0)
            ),
            "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
            "training_seconds": float(time.perf_counter() - started_at),
        }
        mlflow.log_metrics(metrics)

        dataset_frame = features.copy()
        dataset_frame["target"] = target.to_numpy()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Hint: Inferred schema contains integer column",
                category=UserWarning,
            )
            training_dataset = from_pandas(
                dataset_frame,
                targets="target",
                name="iris_training_data",
                digest=dataset_sha256[:12],
            )
            mlflow.log_input(
                training_dataset,
                context="training",
            )

        labels = sorted(int(value) for value in target.unique())
        report = classification_report(
            y_test,
            predictions,
            labels=labels,
            target_names=[IRIS_CLASS_NAMES.get(value, str(value)) for value in labels],
            output_dict=True,
            zero_division=0,
        )
        mlflow.log_dict(report, "evaluation/classification_report.json")
        mlflow.log_dict(
            {
                "labels": labels,
                "class_names": [IRIS_CLASS_NAMES.get(value, str(value)) for value in labels],
                "matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
            },
            "evaluation/confusion_matrix.json",
        )
        mlflow.log_dict(
            {
                "sha256": dataset_sha256,
                "rows": len(features),
                "features": list(IRIS_FEATURE_NAMES),
                "target_distribution": {
                    str(key): int(value)
                    for key, value in target.value_counts().sort_index().items()
                },
            },
            "dataset/profile.json",
        )

        input_example = x_train.iloc[:2]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*Any type hint is inferred as AnyType.*",
                category=UserWarning,
                module="mlflow\\..*",
            )
            warnings.filterwarnings(
                "ignore",
                message="Hint: Inferred schema contains integer column",
                category=UserWarning,
            )
            inferred_inputs = mlflow.models.infer_signature(x_train)
            signature = ModelSignature(
                inputs=inferred_inputs.inputs,
                outputs=Schema([ColSpec(DataType.long)]),
            )
            model_info = mlflow.sklearn.log_model(
                model,
                name="model",
                signature=signature,
                input_example=input_example,
                metadata={
                    "model_version": model_version,
                    "dataset_sha256": dataset_sha256,
                    "source_sha256": source_sha256,
                },
            )
        mlflow.set_tag("logged_model_uri", model_info.model_uri)

        run_id = run.info.run_id

    artifact_dir = Path(config.output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    local_model_dir = artifact_dir / f"model_{model_version}"
    if local_model_dir.exists():
        shutil.rmtree(local_model_dir)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Hint: Inferred schema contains integer column",
            category=UserWarning,
        )
        mlflow.sklearn.save_model(
            sk_model=model,
            path=str(local_model_dir),
            signature=signature,
            input_example=input_example,
            metadata={
                "model_version": model_version,
                "dataset_sha256": dataset_sha256,
                "source_sha256": source_sha256,
            },
        )

    return {
        "run_id": run_id,
        "accuracy": metrics["accuracy"],
        "metrics": metrics,
        "model_uri": model_info.model_uri,
        "feature_names": list(features.columns),
        "classes": [
            {"id": value, "name": IRIS_CLASS_NAMES.get(value, str(value))}
            for value in sorted(np.unique(target).tolist())
        ],
        "model_version": model_version,
        "source": {"sha256": source_sha256, "runtime": runtime_versions},
        "local_model_path": str(local_model_dir),
        "dataset": {
            "name": Path(config.dataset_path).name if config.dataset_path else "sklearn_iris",
            "rows": len(features),
            "sha256": dataset_sha256,
        },
    }
