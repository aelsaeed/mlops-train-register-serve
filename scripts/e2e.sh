#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=${MODEL_NAME:-iris-classifier}
EXPERIMENT_NAME=${MLFLOW_EXPERIMENT_NAME:-mlops-train-register-serve}
MLFLOW_PORT=${MLFLOW_PORT:-5000}
TRACKING_URI=http://127.0.0.1:$MLFLOW_PORT
API_PORT=${API_PORT:-8000}
export MODEL_NAME MLFLOW_PORT API_PORT
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="$ROOT_DIR/artifacts/e2e"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi
mkdir -p "$ARTIFACT_DIR"

if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
else
  echo "[e2e] Docker Compose is required" >&2
  exit 1
fi

cleanup() {
  "${COMPOSE[@]}" down
}
trap cleanup EXIT

"${COMPOSE[@]}" up -d mlflow
for _ in {1..30}; do
  if curl --fail --silent "http://127.0.0.1:$MLFLOW_PORT/health" > /dev/null; then
    break
  fi
  sleep 1
done
curl --fail-with-body --silent --show-error "http://127.0.0.1:$MLFLOW_PORT/health" > /dev/null

export MLFLOW_TRACKING_URI="$TRACKING_URI"
export MLFLOW_EXPERIMENT_NAME="$EXPERIMENT_NAME"

"$PYTHON_BIN" "$ROOT_DIR/train.py" \
  --model-type rf \
  --n-estimators 200 \
  --max-depth 4 \
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

if [[ "${SKIP_API_BUILD:-0}" == "1" ]]; then
  "${COMPOSE[@]}" up -d api
else
  "${COMPOSE[@]}" up -d --build api
fi

for _ in {1..30}; do
  if curl --fail --silent "http://127.0.0.1:$API_PORT/health/ready" > /dev/null; then
    break
  fi
  sleep 1
done
if ! curl --fail-with-body --silent --show-error \
  "http://127.0.0.1:$API_PORT/health/ready" > /dev/null; then
  "${COMPOSE[@]}" logs api
  exit 1
fi
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
cat "$ARTIFACT_DIR/predict_response.json"
