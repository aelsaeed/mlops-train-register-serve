from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient


@dataclass(frozen=True)
class TrackingConfig:
    tracking_uri: Optional[str]
    experiment_name: str
    artifact_location: Optional[str]


def configure_tracking(config: TrackingConfig) -> None:
    if config.tracking_uri:
        mlflow.set_tracking_uri(config.tracking_uri)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(config.experiment_name)
    if experiment is None:
        client.create_experiment(
            name=config.experiment_name,
            artifact_location=config.artifact_location,
        )
    mlflow.set_experiment(config.experiment_name)
