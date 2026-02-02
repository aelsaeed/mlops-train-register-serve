from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mlops_trs.registry import RegistryConfig, register_best_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register the best MLflow run.")
    parser.add_argument(
        "--model-name",
        default=os.getenv("MLFLOW_MODEL_NAME", "iris-classifier"),
    )
    parser.add_argument(
        "--experiment-name",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "mlops-train-register-serve"),
    )
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument(
        "--artifact-location",
        default=os.getenv("MLFLOW_ARTIFACT_LOCATION"),
    )
    parser.add_argument(
        "--output",
        default="artifacts/registered_model.json",
        help="Where to write model version metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    version = register_best_run(
        RegistryConfig(
            model_name=args.model_name,
            experiment_name=args.experiment_name,
            tracking_uri=args.tracking_uri,
            artifact_location=args.artifact_location,
        )
    )
    payload = {
        "name": version.name,
        "version": version.version,
        "run_id": version.run_id,
        "current_stage": version.current_stage,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
