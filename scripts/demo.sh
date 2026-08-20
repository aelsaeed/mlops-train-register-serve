#!/usr/bin/env bash
set -euo pipefail

MODE="dry"
if [[ "${1:-}" == "--full" ]]; then
  MODE="full"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="$ROOT_DIR/artifacts/demo"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi
TRACKING_SQLITE="sqlite:///$ARTIFACT_DIR/mlflow.db"
MODEL_NAME="${MODEL_NAME:-iris-classifier}"
EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-mlops-train-register-serve}"
API_PORT="${API_PORT:-8000}"
mkdir -p "$ARTIFACT_DIR"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

run_dry_demo() {
  echo "[demo] Running dry lifecycle demo..."
  export MLFLOW_TRACKING_URI="$TRACKING_SQLITE"
  export MLFLOW_EXPERIMENT_NAME="$EXPERIMENT_NAME"
  export MLFLOW_MODEL_NAME="$MODEL_NAME"
  export MLFLOW_ARTIFACT_LOCATION="$(
    ARTIFACT_PATH="$ARTIFACT_DIR/artifacts" "$PYTHON_BIN" -c \
      'import os; from pathlib import Path; print(Path(os.environ["ARTIFACT_PATH"]).resolve().as_uri())'
  )"

  "$PYTHON_BIN" "$ROOT_DIR/train.py" \
    --dataset-path "$ROOT_DIR/data/sample.csv" \
    --model-type logreg \
    --output "$ARTIFACT_DIR/train_output.json" \
    --model-output-dir "$ARTIFACT_DIR/models"

  "$PYTHON_BIN" "$ROOT_DIR/scripts/register.py" \
    --model-name "$MODEL_NAME" \
    --run-file "$ARTIFACT_DIR/train_output.json" \
    --output "$ARTIFACT_DIR/registered_model.json"

  "$PYTHON_BIN" "$ROOT_DIR/scripts/promote.py" \
    --model-name "$MODEL_NAME" \
    --version-file "$ARTIFACT_DIR/registered_model.json" \
    --output "$ARTIFACT_DIR/promoted_model.json"

  MODEL_URI="models:/$MODEL_NAME@champion" \
    MLFLOW_TRACKING_URI="$TRACKING_SQLITE" \
    "$PYTHON_BIN" -m uvicorn mlops_trs.api:app --host 127.0.0.1 --port "$API_PORT" \
    > "$ARTIFACT_DIR/api.log" 2>&1 &
  SERVER_PID=$!

  for _ in {1..20}; do
    if curl --fail --silent "http://127.0.0.1:$API_PORT/health/ready" > /dev/null; then
      break
    fi
    sleep 1
  done

  curl --fail-with-body --silent --show-error -X POST "http://127.0.0.1:$API_PORT/predict" \
    -H 'Content-Type: application/json' \
    -d '{"instances": [{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]}' \
    > "$ARTIFACT_DIR/predict_response.json"
  curl --fail-with-body --silent --show-error "http://127.0.0.1:$API_PORT/metrics" \
    > "$ARTIFACT_DIR/metrics.txt"

  bash "$ROOT_DIR/scripts/smoke_test.sh" \
    "$ARTIFACT_DIR/train_output.json" \
    "$ARTIFACT_DIR/predict_response.json" \
    "$ARTIFACT_DIR/metrics.txt"
  echo "[demo] Dry demo complete. Output: $ARTIFACT_DIR/predict_response.json"
}

run_full_demo() {
  if ! docker compose version >/dev/null 2>&1 && ! command -v docker-compose >/dev/null 2>&1; then
    echo "[demo] Docker Compose not found; falling back to --dry mode."
    run_dry_demo
    return
  fi

  echo "[demo] Running full demo (docker-compose + register + promote + serve)..."
  bash "$ROOT_DIR/scripts/e2e.sh"
  echo "[demo] Full demo complete."
}

if [[ "$MODE" == "full" ]]; then
  run_full_demo
else
  run_dry_demo
fi
