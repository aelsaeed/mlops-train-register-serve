from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mlops_trs.training import TrainConfig, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model and log to MLflow.")
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
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI"),
    )
    parser.add_argument(
        "--artifact-location",
        default=os.getenv("MLFLOW_ARTIFACT_LOCATION"),
    )
    parser.add_argument(
        "--output",
        default="artifacts/train_output.json",
        help="Where to write run metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        model_type=args.model_type,
        random_seed=args.random_seed,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        c_value=args.c_value,
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        artifact_location=args.artifact_location,
    )
    result = train_model(config)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
