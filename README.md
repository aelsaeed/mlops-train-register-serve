# MLOps Train, Register, Serve (MLflow)

![CI](https://github.com/aelsaeed/mlops-train-register-serve/actions/workflows/ci.yml/badge.svg)

Compact end-to-end MLOps workflow using MLflow for tracking + registry and FastAPI for serving.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
make setup
```

## Development

```bash
make setup
pre-commit install
```

Run local quality checks:

```bash
make lint
make typecheck
make test
```

## Lifecycle

```text
[data/sample.csv]
      |
      v
  make train  ---> artifacts/train_output.json + artifacts/model_v1-*/
      |
      v
  make register (tag: lifecycle=dev)
      |
      v
  make promote  (tag: lifecycle=staging, promotion=dev-to-staging)
      |
      v
  make serve  ---> /health  /predict  /metrics
      |
      v
  make demo (dry: train+register+promote+predict in one command)
```

## Demo

### 1-minute dry demo (default)

```bash
make demo
```

Dry mode is deterministic and Docker-free:
- trains on `data/sample.csv`,
- registers best run to local MLflow SQLite registry,
- promotes dev → staging,
- starts API and hits `/predict` with a sample payload.

Artifacts are written under `artifacts/demo/`.

### Full demo (docker compose)

```bash
bash scripts/demo.sh --full
```

## MLflow tracking + registry

Start local MLflow server (for UI + remote-style local workflow):

```bash
make mlflow
```

Open: <http://localhost:5000>

### Train

```bash
make train
```

Produces:
- `artifacts/train_output.json`
- local model artifact dir under `artifacts/model_v1-*/`

### Register (dev)

```bash
make register
```

### Promote (dev → staging)

```bash
make promote
```

## Serving

```bash
make serve
```

Endpoints:
- `GET /health`
- `POST /predict`
- `GET /metrics`

Sample request:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

## Testing/CI

Workflow: `.github/workflows/ci.yml`
- lint: `ruff check .`
- typecheck: `mypy .`
- tiny deterministic training run with `data/sample.csv`
- unit tests: `pytest`
- optional compose validation: `docker compose config`

## Troubleshooting

- `make demo` fails due to missing dependencies: run `make setup`.
- Docker unavailable: use default dry mode (`make demo`).
- Port conflict on `8000`: stop conflicting process or run `uvicorn` on a different port.

## Make targets

- `make setup`
- `make lint`
- `make typecheck`
- `make test`
- `make fmt`
- `make mlflow`
- `make train`
- `make register`
- `make promote`
- `make serve`
- `make demo`
- `make clean`
