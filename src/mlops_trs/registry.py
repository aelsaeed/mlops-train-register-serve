from __future__ import annotations

import math
from dataclasses import dataclass

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from mlops_trs.constants import CANDIDATE_ALIAS, CHAMPION_ALIAS
from mlops_trs.mlflow_utils import TrackingConfig, configure_tracking


@dataclass(frozen=True)
class RegistryConfig:
    model_name: str
    experiment_name: str
    tracking_uri: str | None
    artifact_location: str | None
    run_id: str | None = None


class PromotionGateError(ValueError):
    """Raised when a candidate does not meet the deployment policy."""


@dataclass(frozen=True)
class PromotionResult:
    model_version: ModelVersion
    accuracy: float
    minimum_accuracy: float
    alias: str


def _assign_alias(
    client: MlflowClient,
    model_name: str,
    alias: str,
    version: str,
) -> None:
    aliases = client.get_registered_model(model_name).aliases
    previous_version = aliases.get(alias)
    if previous_version and str(previous_version) != str(version):
        remaining_aliases = sorted(
            name
            for name, assigned_version in aliases.items()
            if name != alias and str(assigned_version) == str(previous_version)
        )
        previous_status = ",".join(remaining_aliases) or f"superseded-{alias}"
        client.set_model_version_tag(
            model_name,
            str(previous_version),
            "deployment_status",
            previous_status,
        )
        client.set_model_version_tag(
            model_name,
            str(previous_version),
            "superseded_by_version",
            str(version),
        )
    client.set_registered_model_alias(model_name, alias, version)


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

    if config.run_id:
        best_run = client.get_run(config.run_id)
        if best_run.info.experiment_id != experiment.experiment_id:
            raise ValueError(
                f"Run {config.run_id} does not belong to experiment {config.experiment_name}"
            )
        if best_run.info.status != "FINISHED":
            raise ValueError(f"Run {config.run_id} is not finished")
    else:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["metrics.accuracy DESC", "attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise ValueError("No runs found to register")
        best_run = runs[0]

    model_uri = best_run.data.tags.get("logged_model_uri", f"runs:/{best_run.info.run_id}/model")
    model_version = mlflow.register_model(model_uri, config.model_name)
    _assign_alias(client, config.model_name, CANDIDATE_ALIAS, str(model_version.version))
    client.set_model_version_tag(
        config.model_name, model_version.version, "deployment_status", CANDIDATE_ALIAS
    )
    client.set_model_version_tag(
        config.model_name,
        model_version.version,
        "source_run_id",
        best_run.info.run_id,
    )
    return model_version


def promote_model(
    model_name: str,
    version: str,
    tracking_uri: str | None,
    minimum_accuracy: float = 0.8,
) -> PromotionResult:
    if not 0 <= minimum_accuracy <= 1:
        raise ValueError("minimum_accuracy must be between 0 and 1")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    model_version = client.get_model_version(name=model_name, version=version)
    try:
        candidate = client.get_model_version_by_alias(model_name, CANDIDATE_ALIAS)
    except MlflowException as exc:
        raise PromotionGateError(f"Model {model_name} has no @{CANDIDATE_ALIAS} alias") from exc
    if str(candidate.version) != str(version):
        raise PromotionGateError(
            f"Promotion blocked: model {model_name} version {version} is not @{CANDIDATE_ALIAS}"
        )
    if not model_version.run_id:
        raise PromotionGateError(f"Model {model_name} version {version} has no source run")

    run = client.get_run(model_version.run_id)
    accuracy = run.data.metrics.get("accuracy")
    if accuracy is None:
        raise PromotionGateError(f"Model {model_name} version {version} has no accuracy metric")
    accuracy = float(accuracy)
    if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        client.set_model_version_tag(model_name, version, "promotion_gate", "failed")
        client.set_model_version_tag(
            model_name, version, "promotion_gate_reason", "invalid_accuracy"
        )
        raise PromotionGateError(
            f"Promotion blocked: accuracy for model {model_name} version {version} "
            "must be a finite value between 0 and 1"
        )
    if accuracy < minimum_accuracy:
        client.set_model_version_tag(model_name, version, "promotion_gate", "failed")
        client.set_model_version_tag(
            model_name, version, "promotion_gate_reason", "accuracy_below_threshold"
        )
        raise PromotionGateError(
            f"Promotion blocked: accuracy {accuracy:.4f} is below {minimum_accuracy:.4f}"
        )

    _assign_alias(client, model_name, CHAMPION_ALIAS, version)
    client.set_model_version_tag(model_name, version, "deployment_status", CHAMPION_ALIAS)
    client.set_model_version_tag(model_name, version, "promotion_gate", "passed")
    client.set_model_version_tag(
        model_name, version, "promotion_gate_reason", "accuracy_at_or_above_threshold"
    )
    client.set_model_version_tag(model_name, version, "promotion", "candidate-to-champion")
    return PromotionResult(
        model_version=client.get_model_version(name=model_name, version=version),
        accuracy=accuracy,
        minimum_accuracy=minimum_accuracy,
        alias=CHAMPION_ALIAS,
    )
