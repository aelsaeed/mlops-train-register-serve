from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field

from mlops_trs.constants import IRIS_CLASS_NAMES, IRIS_FEATURE_NAMES

LOGGER = logging.getLogger(__name__)
ModelLoader = Callable[[str], Any]


class IrisFeatures(BaseModel):
    """Named, validated inputs matching the logged MLflow signature."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    sepal_length: float = Field(gt=0, le=10, examples=[5.1])
    sepal_width: float = Field(gt=0, le=10, examples=[3.5])
    petal_length: float = Field(gt=0, le=10, examples=[1.4])
    petal_width: float = Field(gt=0, le=10, examples=[0.2])


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instances: list[IrisFeatures] = Field(min_length=1, max_length=100)


class Prediction(BaseModel):
    class_id: int | str
    class_name: str
    confidence: float | None


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    model_uri: str
    count: int


class HealthResponse(BaseModel):
    status: str


class ModelResponse(BaseModel):
    model_uri: str
    loaded: bool
    feature_names: list[str]


class ModelProvider:
    """Thread-safe lazy loader so liveness does not depend on the registry."""

    def __init__(self, model_uri: str, loader: ModelLoader) -> None:
        self.model_uri = model_uri
        self._loader = loader
        self._model: Any | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def get(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._loader(self.model_uri)
        return self._model


def _load_model(model_uri: str) -> Any:
    return mlflow.sklearn.load_model(model_uri)


def _scalar(value: Any) -> int | str:
    item = value.item() if hasattr(value, "item") else value
    if isinstance(item, (int, float)) and float(item).is_integer():
        return int(item)
    return str(item)


def _class_name(class_id: int | str) -> str:
    if isinstance(class_id, int):
        return IRIS_CLASS_NAMES.get(class_id, str(class_id))
    return class_id


def create_app(model_uri: str, model_loader: ModelLoader | None = None) -> FastAPI:
    app = FastAPI(
        title="MLOps Train, Register, Serve API",
        version="0.2.0",
        description="Typed batch inference backed by an MLflow champion model.",
    )
    provider = ModelProvider(model_uri, model_loader or _load_model)

    registry = CollectorRegistry()
    prediction_requests = Counter(
        "mlops_prediction_requests_total",
        "Prediction requests received by the service.",
        registry=registry,
    )
    prediction_errors = Counter(
        "mlops_prediction_errors_total",
        "Prediction requests that failed during model inference.",
        registry=registry,
    )
    prediction_rows = Counter(
        "mlops_prediction_rows_total",
        "Rows scored by the model.",
        registry=registry,
    )
    prediction_latency = Histogram(
        "mlops_prediction_duration_seconds",
        "End-to-end model inference latency.",
        registry=registry,
    )
    last_success = Gauge(
        "mlops_last_successful_prediction_unixtime",
        "Unix timestamp of the latest successful prediction.",
        registry=registry,
    )

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    @app.get("/health/live", response_model=HealthResponse, tags=["health"])
    def live() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get(
        "/health/ready",
        response_model=HealthResponse,
        responses={503: {"description": "Model is unavailable"}},
        tags=["health"],
    )
    def ready() -> HealthResponse:
        try:
            provider.get()
        except Exception as exc:
            LOGGER.warning("Model readiness check failed for %s", model_uri, exc_info=exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is unavailable",
            ) from exc
        return HealthResponse(status="ready")

    @app.get("/model", response_model=ModelResponse, tags=["model"])
    def model_metadata() -> ModelResponse:
        return ModelResponse(
            model_uri=model_uri,
            loaded=provider.loaded,
            feature_names=list(IRIS_FEATURE_NAMES),
        )

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.post(
        "/predict",
        response_model=PredictResponse,
        responses={503: {"description": "Model inference is unavailable"}},
        tags=["model"],
    )
    def predict(request: PredictRequest) -> PredictResponse:
        prediction_requests.inc()
        started_at = time.perf_counter()
        try:
            model = provider.get()
            frame = pd.DataFrame(
                [instance.model_dump() for instance in request.instances],
                columns=list(IRIS_FEATURE_NAMES),
            )
            raw_predictions = model.predict(frame)
            probabilities = model.predict_proba(frame) if hasattr(model, "predict_proba") else None
            predictions = []
            for index, value in enumerate(raw_predictions):
                class_id = _scalar(value)
                confidence = float(max(probabilities[index])) if probabilities is not None else None
                predictions.append(
                    Prediction(
                        class_id=class_id,
                        class_name=_class_name(class_id),
                        confidence=confidence,
                    )
                )
        except Exception as exc:
            prediction_errors.inc()
            LOGGER.exception("Prediction failed for model %s", model_uri)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model inference is unavailable",
            ) from exc
        finally:
            prediction_latency.observe(time.perf_counter() - started_at)

        prediction_rows.inc(len(predictions))
        last_success.set_to_current_time()
        return PredictResponse(
            predictions=predictions,
            model_uri=model_uri,
            count=len(predictions),
        )

    return app


MODEL_URI = os.getenv("MODEL_URI", "models:/iris-classifier@champion")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
app = create_app(MODEL_URI)
