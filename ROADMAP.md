# Roadmap

This roadmap favors demonstrable lifecycle controls over adding infrastructure for its own sake.
Completed items describe the current reference implementation; planned items are not production
claims.

## Completed: reproducible train-to-serve path

- [x] Package the Python project for editable development installs on Python 3.11 and 3.12.
- [x] Log the dataset hash, model signature, input example, accuracy, macro precision, macro
  recall, macro F1, and confusion matrix to MLflow.
- [x] Reject malformed training schemas, missing or non-finite values, implausible coarse ranges,
  incomplete class sets, and infeasible stratified splits before fitting.
- [x] Replace deprecated registry stages with `candidate` and `champion` aliases.
- [x] Require a configurable minimum-accuracy gate before champion promotion.
- [x] Serve typed batch predictions with named Iris features, class labels, and confidence.
- [x] Separate liveness and readiness checks and expose model metadata.
- [x] Export valid Prometheus request, error, and latency metrics.
- [x] Build a local Docker Compose stack for MLflow and the API.
- [x] Validate linting, typing, tests, training, packaging, and a non-root containerized
  train-to-predict lifecycle across the CI workflow.
- [x] Audit Python dependencies in CI with a documented, narrow temporary exception.
- [x] Document intended use and limitations in a model card.

## Next: evaluation and data confidence

- [ ] Separate the tiny CI fixture from a statistically useful evaluation dataset.
- [ ] Add cross-validation and a baseline comparison with confidence intervals.
- [ ] Measure probability calibration before treating model probability as decision confidence.
- [ ] Replace coarse feature bounds with dataset-specific validation and a fuller class-balance
  policy.
- [ ] Make the promotion policy consider macro F1 and per-class regressions in addition to
  accuracy.
- [ ] Publish versioned evaluation reports as CI artifacts.

## Next: observability and operations

- [ ] Collect prediction distributions and data-quality signals without retaining sensitive raw
  payloads.
- [ ] Add a local Prometheus and Grafana profile with a documented dashboard and alert thresholds.
- [ ] Define rollback behavior and verify that a rejected candidate cannot displace the champion.
- [ ] Add a controlled model-refresh or rollout mechanism after the `champion` alias moves.
- [ ] Add structured logs, correlation IDs, and an incident-response runbook.
- [ ] Expand security checks to container, secret, and software-bill-of-materials scanning.
- [ ] Remove the temporary `PYSEC-2026-3552` audit exception when MLflow supports
  `cryptography>=50`.
- [ ] Publish signed semantic-version releases with reproducible images.

## Later: production topology

- [ ] Move MLflow metadata and artifacts to managed PostgreSQL and object storage.
- [ ] Add authentication, authorization, TLS, backup, and retention policies.
- [ ] Deploy an immutable image to a managed runtime with autoscaling and rollout controls.
- [ ] Add scheduled retraining, drift-triggered evaluation, and human approval where appropriate.
- [ ] Exercise canary promotion and rollback against realistic traffic and failure scenarios.

See [MODEL_CARD.md](MODEL_CARD.md) for model-specific risks and
[README.md](README.md#design-choices-and-limitations) for current system tradeoffs.
