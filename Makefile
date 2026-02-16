PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.DEFAULT_GOAL := demo

.PHONY: setup lint typecheck test fmt demo clean smoke mlflow train register promote serve

setup:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt -r requirements-dev.txt
	pre-commit install

lint:
	PYTHONPATH=src:. ruff check .

fmt:
	PYTHONPATH=src:. ruff check --fix .

# keep typecheck separate from lint for clearer CI failures
typecheck:
	PYTHONPATH=src:. mypy .

test:
	PYTHONPATH=src:. pytest

smoke:
	bash scripts/smoke_test.sh

# 1-minute local path: tiny train -> register -> promote -> serve -> predict
demo:
	bash scripts/demo.sh --dry

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf artifacts/demo artifacts/smoke artifacts/model_v1-*

mlflow:
	docker compose up -d mlflow

train:
	PYTHONPATH=src:. $(PYTHON) train.py --dataset-path data/sample.csv --output artifacts/train_output.json

register:
	PYTHONPATH=src:. $(PYTHON) scripts/register.py

promote:
	PYTHONPATH=src:. $(PYTHON) scripts/promote.py

serve:
	MODEL_URI=$${MODEL_URI:-models:/iris-classifier/Staging} \
		MLFLOW_TRACKING_URI=$${MLFLOW_TRACKING_URI:-sqlite:///artifacts/demo/mlflow.db} \
		PYTHONPATH=src:. $(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000
