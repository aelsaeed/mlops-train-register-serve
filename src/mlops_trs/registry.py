from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

from mlops_trs.mlflow_utils import TrackingConfig, configure_tracking


@dataclass(frozen=True)
class RegistryConfig:
    model_name: str
    experiment_name: str
    tracking_uri: Optional[str]
    artifact_location: Optional[str]


def register_best_run(config: RegistryConfig) -> ModelVersion:
    configure_tracking(
        TrackingConfig(
            tracking_uri=config.tracking_uri,
            experiment_name=config.experiment_name,
            artifact_location=config.artifact_location,
        )
    )

    client = MlflowClient()
    experiment = client.get_experiment_by_name(config.experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment {config.experiment_name} not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=["metrics.accuracy DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError("No runs found to register")

    best_run = runs[0]
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model_version = mlflow.register_model(model_uri, config.model_name)
    return model_version


def promote_model(
    model_name: str,
    version: str,
    tracking_uri: Optional[str],
) -> ModelVersion:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging",
        archive_existing_versions=True,
    )
    return client.get_model_version(name=model_name, version=version)
