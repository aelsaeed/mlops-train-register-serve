# Roadmap

## Milestone 1: Stabilize demo
- Guarantee a 1-minute local evaluation path via `make demo` dry mode.
- Keep full MLflow registry path available through `scripts/demo.sh --full`.
- Expand smoke checks to validate output contracts and API readiness.

## Milestone 2: Observability + metrics
- Add service-level metrics (latency, throughput, error rate) for serving path.
- Track model drift and prediction distributions over time.
- Publish baseline dashboards and alert thresholds.

## Milestone 3: Hardening + docs
- Add release process and semantic versioning policy.
- Add security checks and dependency vulnerability scanning in CI.
- Document production deployment topology and incident response playbooks.
