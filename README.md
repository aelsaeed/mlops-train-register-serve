# MLOps Train, Register, Serve (MLflow)

A compact end-to-end MLOps workflow built on MLflow. It trains a model, tracks runs in a
local MLflow server, registers the best run in the MLflow Model Registry, promotes it to
Staging, and serves predictions via MLflow or a FastAPI wrapper.

## Quickstart

```bash
make setup
make mlflow
make train
make register
python scripts/promote.py
```

## Training

Run a training job and log params/metrics/artifacts to MLflow:

```bash
python train.py --model-type rf --n-estimators 200 --max-depth 4
```

Arguments:
- `--model-type`: `logreg` or `rf`
- `--random-seed`: RNG seed
- `--test-size`: test split fraction
- `--n-estimators`: RF trees (ignored by logreg)
- `--max-depth`: RF max depth (ignored by logreg)
- `--c-value`: Logistic regression C value (ignored by RF)

Outputs run metadata to `artifacts/train_output.json`.

## MLflow Tracking UI

Start the local MLflow tracking server (SQLite backend + local artifact store):

```bash
make mlflow
```

Open [http://localhost:5000](http://localhost:5000) to explore runs.

## Register the Best Model

Register the best run (by accuracy) into the MLflow Model Registry:

```bash
python scripts/register.py --model-name iris-classifier
```

This writes `artifacts/registered_model.json` with the model version info.

## Promote to Staging

Promote the model version to Staging with a script (no manual clicks):

```bash
python scripts/promote.py --model-name iris-classifier
```

## Serve the Model

### Option A: MLflow model serving

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 \
mlflow models serve -m models:/iris-classifier/Staging -p 5001
```

Example request:

```bash
curl -X POST http://localhost:5001/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_records": [[5.1, 3.5, 1.4, 0.2]]}'
```

### Option B: FastAPI wrapper

```bash
MODEL_URI=models:/iris-classifier/Staging \
MLFLOW_TRACKING_URI=http://localhost:5000 \
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

## End-to-End Script

Run training → register → promote → serve → smoke test:

```bash
scripts/e2e.sh
```

## Make Targets

- `make setup`: install dependencies
- `make mlflow`: start MLflow tracking server via Docker Compose
- `make train`: run training
- `make register`: register the best run
- `make serve`: start FastAPI server
- `make test`: run pytest
- `make lint`: run ruff and mypy

## Production Notes

- **Secrets**: store credentials (databases, artifact storage, etc.) in a secrets manager
  and inject via environment variables rather than committing to source control.
- **Model versioning**: keep semantic model versions or tags in the MLflow Model Registry
  and automate promotion gates with CI/CD pipelines.
- **Canary deployments**: route a small percentage of traffic to new model versions and
  observe metrics before full rollout.
- **Monitoring**: log prediction distributions, latency, and data drift indicators; alert
  on SLA violations and anomalous behavior.

## Case Study

**Constraints**
- Local-only stack for reproducibility (SQLite backend, local artifact store).
- Minimal dataset and fast training loop for CI-friendly runtimes.
- Two serving paths to illustrate both MLflow-native and custom API serving.

**Tradeoffs**
- Simplicity over scalability (single-node MLflow, no external object store).
- Lightweight tests over exhaustive evaluation metrics.
- Fixed dataset schema for predictable inference contracts.

**Results**
- Synthetic run with RandomForest (n_estimators=200, max_depth=4) achieved ~0.95 accuracy.
- End-to-end pipeline completes in seconds locally and produces a staged model version.
