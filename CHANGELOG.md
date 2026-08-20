# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- MLflow 3 model-registry workflow with `candidate` and `champion` aliases.
- Configurable minimum-accuracy promotion gate that preserves the current champion on failure.
- Dataset SHA-256, model signature, input example, macro precision, macro recall, macro F1, and
  confusion-matrix logging alongside accuracy.
- Pre-training validation for schema, finite feature values, coarse input bounds, required Iris
  classes, and feasible stratified splits.
- Source and key-runtime fingerprints that prevent tracked code or dependency changes from
  silently reusing a local model identifier.
- Typed batch prediction contract with named Iris features, class labels, and confidence values.
- Liveness, readiness, model-metadata, and Prometheus metrics endpoints.
- Installable Python package with editable development dependencies on Python 3.11 and 3.12.
- Tested dependency constraints shared by local setup, package builds, and CI.
- Containerized API and local MLflow services in Docker Compose.
- CI dependency audit with a documented temporary exception for `PYSEC-2026-3552` until MLflow
  permits `cryptography>=50`.
- Registry, promotion-gate, API-contract, health, and metrics coverage in the automated test suite.
- Case-study README, model card, updated roadmap, and expanded contribution guidance.

### Changed

- Model serving now resolves `models:/iris-classifier@champion` instead of a registry stage.
- Training output now carries richer evaluation and lineage metadata.
- Local setup now installs the package rather than relying on an ad hoc `PYTHONPATH`.
- CI validates Python 3.11 and 3.12, coverage, the deterministic training path, package builds,
  dependencies, and a non-root containerized train-to-predict lifecycle.
- The demo and documentation now distinguish a small lifecycle fixture from a meaningful model
  benchmark.

### Removed

- Deprecated MLflow `Staging` stage transitions and stage-based model URIs.
- Positional single-row prediction payloads.

## [0.1.0] - 2026-02-16

### Added

- Initial MLflow training, tracking, model registration, and promotion workflow.
- FastAPI prediction service with basic health and metrics endpoints.
- Deterministic local demo, smoke checks, unit tests, Ruff, mypy, pre-commit, and GitHub Actions.
- Docker Compose MLflow service, contribution templates, roadmap, CODEOWNERS, and MIT license.
