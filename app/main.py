from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

PREDICT_REQUESTS = 0
LAST_INFERENCE_UNIX = 0.0


class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_length=4, max_length=4)


class PredictResponse(BaseModel):
    predictions: list[int]
    model_uri: str


def _load_model(model_uri: str) -> Any:
    return mlflow.pyfunc.load_model(model_uri)


@lru_cache(maxsize=1)
def get_model(model_uri: str) -> Any:
    return _load_model(model_uri)


def create_app(model_uri: str) -> FastAPI:
    app = FastAPI(title="MLflow FastAPI Serving")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics() -> str:
        return (
            "# TYPE app_predict_requests_total counter\n"
            f"app_predict_requests_total {PREDICT_REQUESTS}\n"
            "# TYPE app_last_inference_unix gauge\n"
            f"app_last_inference_unix {LAST_INFERENCE_UNIX}\n"
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        global PREDICT_REQUESTS, LAST_INFERENCE_UNIX

        model = get_model(model_uri)
        frame = pd.DataFrame([request.features])
        predictions = model.predict(frame)

        PREDICT_REQUESTS += 1
        LAST_INFERENCE_UNIX = time.time()
        return PredictResponse(predictions=[int(predictions[0])], model_uri=model_uri)

    return app


MODEL_URI = os.getenv("MODEL_URI", "models:/iris-classifier/Staging")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
app = create_app(MODEL_URI)
