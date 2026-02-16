#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_PATH="${1:-$ROOT_DIR/artifacts/demo/train_output.json}"
SMOKE_DIR="$ROOT_DIR/artifacts/smoke"
mkdir -p "$SMOKE_DIR"

if [[ ! -f "$OUTPUT_PATH" ]]; then
  echo "[smoke] missing expected output: $OUTPUT_PATH" >&2
  exit 1
fi

OUTPUT_PATH="$OUTPUT_PATH" python3 - <<'PY'
import json
import os
from pathlib import Path

output_path = Path(os.environ["OUTPUT_PATH"])
data = json.loads(output_path.read_text())
required = {"run_id", "model_uri", "accuracy", "model_version", "local_model_path"}
missing = sorted(required - data.keys())
if missing:
    raise SystemExit(f"[smoke] missing keys in {output_path}: {missing}")
if not str(data["model_version"]).startswith("v1-"):
    raise SystemExit("[smoke] model_version does not start with v1-")
if not Path(data["local_model_path"]).exists():
    raise SystemExit("[smoke] local_model_path does not exist")
print("[smoke] output JSON contains required deterministic model metadata")
PY

PYTHONPATH=src:. pytest -q tests/test_api.py tests/test_training.py

echo "[smoke] success"
