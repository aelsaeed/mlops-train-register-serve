# syntax=docker/dockerfile:1

FROM python:3.14-slim@sha256:ce40764625a4ff50df3548277632e7f96c4e77fe75fa848aae9885476e7df5a4 AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

COPY pyproject.toml constraints.txt README.md LICENSE ./
COPY src ./src

RUN python -m pip install --constraint constraints.txt setuptools==82.0.1 build \
    && python -m build --wheel --no-isolation --outdir /wheels


FROM python:3.14-slim@sha256:ce40764625a4ff50df3548277632e7f96c4e77fe75fa848aae9885476e7df5a4 AS runtime

LABEL org.opencontainers.image.source="https://github.com/aelsaeed/mlops-train-register-serve" \
      org.opencontainers.image.title="MLOps Train, Register, Serve API" \
      org.opencontainers.image.description="Non-root FastAPI serving image for an MLflow champion model"

ENV HOME=/home/mlops \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI="http://mlflow:5000" \
    MODEL_URI="models:/iris-classifier@champion"

WORKDIR /app

COPY constraints.txt /tmp/constraints.txt
COPY --from=builder /wheels /tmp/wheels

RUN python -m pip install --constraint /tmp/constraints.txt /tmp/wheels/*.whl \
    && useradd --create-home --uid 10001 --user-group --shell /usr/sbin/nologin mlops \
    && rm -rf /tmp/wheels /tmp/constraints.txt

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2).read()"]

CMD ["mlops-serve", "--host", "0.0.0.0", "--port", "8000"]
