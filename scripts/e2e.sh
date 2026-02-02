#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=${MODEL_NAME:-iris-classifier}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-mlops-train-register-serve}
TRACKING_URI=${MLFLOW_TRACKING_URI:-http://localhost:5000}

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" || true
  fi
  docker compose down
}
trap cleanup EXIT

docker compose up -d mlflow
sleep 5

export MLFLOW_TRACKING_URI="$TRACKING_URI"
export MLFLOW_EXPERIMENT_NAME="$EXPERIMENT_NAME"
mkdir -p artifacts

python train.py --model-type rf --n-estimators 200 --max-depth 4
python scripts/register.py --model-name "$MODEL_NAME"
python scripts/promote.py --model-name "$MODEL_NAME"

MODEL_URI="models:/$MODEL_NAME/Staging" \
  python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

sleep 3
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}' > artifacts/smoke_response.json
cat artifacts/smoke_response.json
