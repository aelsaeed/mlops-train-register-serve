"""ASGI compatibility module for ``uvicorn app.main:app``."""

from mlops_trs.api import app, create_app

__all__ = ["app", "create_app"]
