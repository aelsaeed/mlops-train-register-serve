#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_PATH="${1:-$ROOT_DIR/artifacts/demo/train_output.json}"
PREDICTION_PATH="${2:-$ROOT_DIR/artifacts/demo/predict_response.json}"
METRICS_PATH="${3:-$ROOT_DIR/artifacts/demo/metrics.txt}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

for expected_file in "$OUTPUT_PATH" "$PREDICTION_PATH" "$METRICS_PATH"; do
  if [[ ! -f "$expected_file" ]]; then
    echo "[smoke] missing expected output: $expected_file" >&2
    exit 1
  fi
done

OUTPUT_PATH="$OUTPUT_PATH" PREDICTION_PATH="$PREDICTION_PATH" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

output_path = Path(os.environ["OUTPUT_PATH"])
data = json.loads(output_path.read_text())
required = {
    "run_id",
    "model_uri",
    "accuracy",
    "metrics",
    "dataset",
    "model_version",
    "local_model_path",
}
missing = sorted(required - data.keys())
if missing:
    raise SystemExit(f"[smoke] missing keys in {output_path}: {missing}")
if not str(data["model_version"]).startswith("v2-"):
    raise SystemExit("[smoke] model_version does not start with v2-")
if not Path(data["local_model_path"]).exists():
    raise SystemExit("[smoke] local_model_path does not exist")

prediction_path = Path(os.environ["PREDICTION_PATH"])
prediction = json.loads(prediction_path.read_text())
if prediction.get("count") != 1 or not prediction.get("predictions"):
    raise SystemExit("[smoke] prediction response violates the API contract")
if prediction["predictions"][0].get("class_name") is None:
    raise SystemExit("[smoke] prediction is missing class_name")
print("[smoke] model metadata and prediction contracts are valid")
PY

if ! grep -q '^mlops_prediction_requests_total ' "$METRICS_PATH"; then
  echo "[smoke] Prometheus request counter is missing" >&2
  exit 1
fi

echo "[smoke] success"
