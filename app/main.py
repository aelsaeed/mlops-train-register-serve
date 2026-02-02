from __future__ import annotations

import os
from functools import lru_cache
from typing import List

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    features: List[float] = Field(..., min_length=4, max_length=4)


class PredictResponse(BaseModel):
    predictions: List[int]
    model_uri: str


def _load_model(model_uri: str) -> mlflow.pyfunc.PyFuncModel:
    return mlflow.pyfunc.load_model(model_uri)


@lru_cache(maxsize=1)
def get_model(model_uri: str) -> mlflow.pyfunc.PyFuncModel:
    return _load_model(model_uri)


def create_app(model_uri: str) -> FastAPI:
    app = FastAPI(title="MLflow FastAPI Serving")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        model = get_model(model_uri)
        frame = pd.DataFrame([request.features])
        predictions = model.predict(frame)
        return PredictResponse(predictions=[int(predictions[0])], model_uri=model_uri)

    return app


MODEL_URI = os.getenv("MODEL_URI", "models:/iris-classifier/Staging")
app = create_app(MODEL_URI)
