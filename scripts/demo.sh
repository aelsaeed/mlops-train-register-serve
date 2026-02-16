#!/usr/bin/env bash
set -euo pipefail

MODE="dry"
if [[ "${1:-}" == "--full" ]]; then
  MODE="full"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="$ROOT_DIR/artifacts/demo"
TRACKING_SQLITE="sqlite:///$ARTIFACT_DIR/mlflow.db"
MODEL_NAME="${MODEL_NAME:-iris-classifier}"
EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-mlops-train-register-serve}"
mkdir -p "$ARTIFACT_DIR"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

run_dry_demo() {
  echo "[demo] Running dry lifecycle demo..."
  export MLFLOW_TRACKING_URI="$TRACKING_SQLITE"
  export MLFLOW_EXPERIMENT_NAME="$EXPERIMENT_NAME"
  export MLFLOW_MODEL_NAME="$MODEL_NAME"
  export MLFLOW_ARTIFACT_LOCATION="$(python3 -c 'from pathlib import Path; print(Path("'$ARTIFACT_DIR'/artifacts").resolve().as_uri())')"

  PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR" python3 "$ROOT_DIR/train.py" \
    --dataset-path "$ROOT_DIR/data/sample.csv" \
    --model-type logreg \
    --output "$ARTIFACT_DIR/train_output.json"

  PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR" python3 "$ROOT_DIR/scripts/register.py" \
    --model-name "$MODEL_NAME" \
    --output "$ARTIFACT_DIR/registered_model.json"

  PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR" python3 "$ROOT_DIR/scripts/promote.py" \
    --model-name "$MODEL_NAME" \
    --version-file "$ARTIFACT_DIR/registered_model.json" \
    > "$ARTIFACT_DIR/promoted_model.json"

  MODEL_URI="models:/$MODEL_NAME/Staging" \
    MLFLOW_TRACKING_URI="$TRACKING_SQLITE" \
    PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR" \
    python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 >/tmp/demo_uvicorn.log 2>&1 &
  SERVER_PID=$!

  for _ in {1..20}; do
    if curl -sf http://localhost:8000/health > /dev/null; then
      break
    fi
    sleep 1
  done

  curl -sf -X POST http://localhost:8000/predict \
    -H 'Content-Type: application/json' \
    -d '{"features": [5.1, 3.5, 1.4, 0.2]}' > "$ARTIFACT_DIR/predict_response.json"
  curl -sf http://localhost:8000/metrics > "$ARTIFACT_DIR/metrics.txt"

  bash "$ROOT_DIR/scripts/smoke_test.sh" "$ARTIFACT_DIR/train_output.json"
  echo "[demo] Dry demo complete. Output: $ARTIFACT_DIR/predict_response.json"
}

run_full_demo() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "[demo] Docker not found; falling back to --dry mode."
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
