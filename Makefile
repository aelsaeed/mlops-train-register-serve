PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
COMPOSE ?= $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")

.DEFAULT_GOAL := help

.PHONY: help setup lint format-check typecheck test coverage check fmt demo clean smoke mlflow train register promote serve build

help:
	@echo "setup         Install the package and development tools"
	@echo "check         Run formatting, lint, types, and tests"
	@echo "demo          Run train -> register -> gate -> serve -> predict locally"
	@echo "mlflow        Start the local MLflow UI with Docker Compose"
	@echo "serve         Serve the current @champion model"
	@echo "build         Build wheel and source distribution"

setup:
	$(PIP) install --upgrade pip
	$(PIP) install -c constraints.txt -e ".[dev]"
	$(PYTHON) -m pre_commit install

lint:
	$(PYTHON) -m ruff check .

format-check:
	$(PYTHON) -m ruff format --check .

fmt:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

# keep typecheck separate from lint for clearer CI failures
typecheck:
	$(PYTHON) -m mypy .

test:
	$(PYTHON) -m pytest -q

coverage:
	$(PYTHON) -m pytest --cov --cov-report=term-missing

check: format-check lint typecheck coverage

smoke:
	bash scripts/smoke_test.sh

# 1-minute local path: tiny train -> register -> promote -> serve -> predict
demo:
	PYTHON_BIN=$(PYTHON) bash scripts/demo.sh --dry

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf artifacts/demo artifacts/e2e artifacts/smoke artifacts/model_v1-* artifacts/model_v2-*

mlflow:
	$(COMPOSE) up -d mlflow

train:
	$(PYTHON) train.py --dataset-path data/sample.csv --output artifacts/train_output.json

register:
	$(PYTHON) scripts/register.py --run-file artifacts/train_output.json

promote:
	$(PYTHON) scripts/promote.py

serve:
	MODEL_URI=$${MODEL_URI:-models:/iris-classifier@champion} \
		MLFLOW_TRACKING_URI=$${MLFLOW_TRACKING_URI:-sqlite:///artifacts/demo/mlflow.db} \
		$(PYTHON) -m uvicorn mlops_trs.api:app --host 0.0.0.0 --port $${API_PORT:-8000}

build:
	$(PYTHON) -m build
