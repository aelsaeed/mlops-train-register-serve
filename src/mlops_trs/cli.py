from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import uvicorn

from mlops_trs.registry import RegistryConfig, promote_model, register_best_run
from mlops_trs.training import TrainConfig, train_model


def _write_json(payload: dict[str, object], output: str | None = None) -> None:
    rendered = json.dumps(payload, indent=2)
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n")
    print(rendered)


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a classifier with MLflow.")
    parser.add_argument("--model-type", choices=["logreg", "rf"], default="logreg")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument(
        "--experiment-name",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "mlops-train-register-serve"),
    )
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--artifact-location", default=os.getenv("MLFLOW_ARTIFACT_LOCATION"))
    parser.add_argument("--dataset-path", default="data/sample.csv")
    parser.add_argument(
        "--model-output-dir",
        default=os.getenv("MODEL_OUTPUT_DIR", "artifacts"),
        help="Directory for the locally exported MLflow model.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/train_output.json",
        help="Where to write run metadata.",
    )
    args = parser.parse_args()

    result = train_model(
        TrainConfig(
            model_type=args.model_type,
            random_seed=args.random_seed,
            test_size=args.test_size,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            c_value=args.c_value,
            experiment_name=args.experiment_name,
            tracking_uri=args.tracking_uri,
            artifact_location=args.artifact_location,
            dataset_path=args.dataset_path,
            output_dir=args.model_output_dir,
        )
    )
    _write_json(result, args.output)


def register_main() -> None:
    parser = argparse.ArgumentParser(description="Register an evaluated MLflow run.")
    parser.add_argument("--model-name", default=os.getenv("MLFLOW_MODEL_NAME", "iris-classifier"))
    parser.add_argument(
        "--experiment-name",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "mlops-train-register-serve"),
    )
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--artifact-location", default=os.getenv("MLFLOW_ARTIFACT_LOCATION"))
    parser.add_argument("--run-id", help="Exact MLflow run to register.")
    parser.add_argument(
        "--run-file",
        default="artifacts/train_output.json",
        help="Training JSON containing a run_id when --run-id is omitted.",
    )
    parser.add_argument("--output", default="artifacts/registered_model.json")
    args = parser.parse_args()

    run_id = args.run_id
    run_file = Path(args.run_file)
    if run_id is None and run_file.is_file():
        run_id = str(json.loads(run_file.read_text())["run_id"])
    version = register_best_run(
        RegistryConfig(
            model_name=args.model_name,
            experiment_name=args.experiment_name,
            tracking_uri=args.tracking_uri,
            artifact_location=args.artifact_location,
            run_id=run_id,
        )
    )
    _write_json(
        {
            "name": version.name,
            "version": version.version,
            "run_id": version.run_id,
            "alias": "candidate",
            "model_uri": f"models:/{version.name}@candidate",
        },
        args.output,
    )


def promote_main() -> None:
    parser = argparse.ArgumentParser(description="Promote a candidate model to champion.")
    parser.add_argument("--model-name", default=os.getenv("MLFLOW_MODEL_NAME", "iris-classifier"))
    parser.add_argument("--version", help="Model version to promote.")
    parser.add_argument("--version-file", default="artifacts/registered_model.json")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument(
        "--minimum-accuracy",
        type=float,
        default=float(os.getenv("MINIMUM_ACCURACY", "0.80")),
    )
    parser.add_argument("--output", default="artifacts/promoted_model.json")
    args = parser.parse_args()

    version = args.version
    if version is None:
        version = str(json.loads(Path(args.version_file).read_text())["version"])
    result = promote_model(
        model_name=args.model_name,
        version=version,
        tracking_uri=args.tracking_uri,
        minimum_accuracy=args.minimum_accuracy,
    )
    _write_json(
        {
            "name": result.model_version.name,
            "version": result.model_version.version,
            "alias": result.alias,
            "model_uri": f"models:/{result.model_version.name}@{result.alias}",
            "promotion": "candidate-to-champion",
            "quality_gate": {
                "metric": "accuracy",
                "value": result.accuracy,
                "minimum": result.minimum_accuracy,
                "status": "passed",
            },
        },
        args.output,
    )


def serve_main() -> None:
    parser = argparse.ArgumentParser(description="Serve the MLflow champion through FastAPI.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("mlops_trs.api:app", host=args.host, port=args.port)
