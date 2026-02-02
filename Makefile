PYTHON ?= python3

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

mlflow:
	docker compose up -d mlflow

train:
	$(PYTHON) train.py

register:
	$(PYTHON) scripts/register.py

serve:
	MODEL_URI=${MODEL_URI:-models:/iris-classifier/Staging} \
		MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://localhost:5000} \
		$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m mypy .
