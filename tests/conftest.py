from __future__ import annotations

from collections.abc import Iterator

import mlflow
import pytest


@pytest.fixture(autouse=True)
def restore_mlflow_tracking_state() -> Iterator[None]:
    """Keep MLflow's process-global fluent state from leaking between tests."""
    previous_tracking_uri = mlflow.get_tracking_uri()
    mlflow.end_run()
    try:
        yield
    finally:
        mlflow.end_run()
        mlflow.set_tracking_uri(previous_tracking_uri)
