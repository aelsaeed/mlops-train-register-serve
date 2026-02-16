from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mlops_trs.registry import promote_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a model version to Staging.")
    parser.add_argument(
        "--model-name",
        default=os.getenv("MLFLOW_MODEL_NAME", "iris-classifier"),
    )
    parser.add_argument("--version", help="Model version to promote.")
    parser.add_argument(
        "--version-file",
        default="artifacts/registered_model.json",
        help="JSON file with a 'version' field.",
    )
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    version = args.version
    if version is None:
        data = json.loads(Path(args.version_file).read_text())
        version = str(data["version"])

    model_version = promote_model(
        model_name=args.model_name,
        version=version,
        tracking_uri=args.tracking_uri,
    )
    payload = {
        "name": model_version.name,
        "version": model_version.version,
        "current_stage": model_version.current_stage,
        "lifecycle_tag": "staging",
        "promotion": "dev-to-staging",
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
