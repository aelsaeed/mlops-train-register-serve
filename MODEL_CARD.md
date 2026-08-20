# Model Card: Iris Classifier

Last updated: 2026-08-20

## Summary

This model is a small multiclass Iris classifier used to demonstrate an end-to-end MLOps
lifecycle. The primary deliverable is the surrounding engineering workflow—tracked training,
evaluation evidence, registry governance, controlled promotion, typed serving, and operational
signals—not a novel or production-qualified model.

| Item | Value |
| --- | --- |
| Registered model | `iris-classifier` |
| Task | Three-class classification |
| Classes | `setosa`, `versicolor`, `virginica` |
| Candidate estimators | Logistic regression and random forest |
| Frameworks | scikit-learn and MLflow 3 |
| Serving reference | `models:/iris-classifier@champion` |
| Inputs | Four numeric Iris measurements |
| Output | Class ID, class name, and model-reported confidence |

MLflow assigns immutable model versions. The mutable `candidate` alias identifies the version
under evaluation, while `champion` identifies the version approved for serving.

## Intended use

- Demonstrating experiment lineage, model registration, automated quality gates, and alias-based
  deployment.
- Exercising a typed batch inference API, health checks, model metadata, and Prometheus metrics.
- Providing a fast, deterministic example for local development, CI, interviews, and technical
  discussion.
- Testing changes to the project's training, registry, and serving contracts.

## Out-of-scope use

- Real botanical identification, scientific research, or ecological decisions.
- Medical, financial, safety-critical, or other high-impact decisions.
- Predictions for plant species or measurements outside the Iris schema and training
  distribution.
- Using the returned probability as a calibrated measure of certainty.
- Assessing production scalability, security, fairness, or reliability from this local example.

## Data

The schema follows the canonical
[Iris dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset):

| Feature | Type | Meaning |
| --- | --- | --- |
| `sepal_length` | Float | Sepal length in centimeters |
| `sepal_width` | Float | Sepal width in centimeters |
| `petal_length` | Float | Petal length in centimeters |
| `petal_width` | Float | Petal width in centimeters |
| `target` | Integer | Class ID: 0, 1, or 2 |

The checked-in [`data/sample.csv`](data/sample.csv) contains 12 labeled rows. It is intentionally
small so the complete lifecycle remains fast in CI; it is not large enough to support a reliable
generalization claim. Training can accept another schema-compatible CSV through the dataset-path
configuration.

Each run logs a SHA-256 hash of the canonicalized feature and target content. The hash provides
lineage and change detection, but it does not establish data quality, provenance, licensing, or
representativeness.

## Training and model selection

Training uses a seeded, stratified train/test split and supports logistic regression and random
forest estimators. MLflow records the estimator configuration, random seed, dataset path and hash,
model signature, input example, metrics, and artifacts.

The standard CLI registers the exact run ID emitted by training and assigns its immutable model
version the `candidate` alias. The registry function can fall back to the highest-accuracy finished
run when no run ID is supplied. Promotion compares the candidate's recorded accuracy with the
configured `MINIMUM_ACCURACY` value. Only a passing candidate receives the `champion` alias; a
failed gate leaves the existing champion unchanged.

Accuracy is intentionally used as a simple, auditable teaching policy. A production policy would
normally combine multiple metrics, data checks, operational evidence, review, and rollback
criteria.

## Evaluation

Every training run logs:

- accuracy;
- macro-averaged precision;
- macro-averaged recall;
- macro-averaged F1;
- a confusion-matrix artifact;
- a classification report with per-class support; and
- separate dataset, source, and runtime lineage metadata.

No fixed benchmark is asserted in this document. Results depend on the selected estimator,
parameters, seed, dependency versions, and dataset. Inspect the MLflow run associated with the
registered model version for the actual evidence used in promotion.

The tiny repository fixture can yield unstable and deceptively strong scores because only a few
observations are evaluated. Before making any quality claim, use a larger representative dataset,
cross-validation or an external test set, class-level error review, and uncertainty estimates.

## Serving behavior

The API accepts one or more observations with named features. It rejects missing, extra, or
malformed values through the typed request contract. A successful prediction contains:

- the numeric class ID;
- the corresponding Iris class name;
- the estimator's highest predicted probability as `confidence`;
- the model URI used for the batch.

The confidence value is not calibrated. Consumers must not interpret, for example, `0.90` as a
guaranteed 90 percent chance of correctness.

`GET /health/live` reports process liveness. `GET /health/ready` verifies that the configured model
can be loaded. `GET /model` exposes the model URI, load state, and feature order. These endpoints
support operation of the example but are not a complete service-level objective.

The serving process caches the model after loading it. Reassigning the `champion` alias requires a
service restart or rollout before an existing process serves the new version.

## Limitations and risks

- Iris is a small, clean, balanced teaching dataset and does not represent real data complexity.
- The 12-row fixture is optimized for pipeline speed, not statistical power.
- Only two classical estimator families are compared; there is no comprehensive search.
- Probabilities are not calibrated and out-of-distribution inputs are not detected.
- Coarse input bounds (`> 0` and `<= 10`) do not establish plausible biological ranges or
  detect out-of-distribution observations.
- No subgroup or fairness analysis is provided; the dataset has no documented human demographic
  attributes and must not be repurposed for human-impact decisions.
- Prometheus metrics cover service behavior, not data drift or long-term model quality.
- Alias reassignment does not trigger a hot reload in an already-running API process.
- The local MLflow and Docker Compose topology does not provide production authentication,
  authorization, encryption, high availability, backups, or autoscaling.

## Monitoring recommendations

For any deployment beyond this reference project:

1. Validate feature schema, ranges, missingness, and distribution changes.
2. Monitor latency, errors, traffic, readiness, class distribution, and confidence distribution.
3. Compare delayed labels with predictions and track per-class performance over time.
4. Alert on service-level objectives and statistically meaningful drift thresholds.
5. Retain privacy-safe lineage connecting a prediction service version to its MLflow run and
   dataset hash.
6. Define human review, rollback, and retraining policies before accepting traffic.

## Reproducibility and review

To reproduce the reference lifecycle, follow the [README quickstart](README.md#quickstart). Record
the source revision, automated source hash, Python and dependency versions, dataset hash, seed,
MLflow run ID, registered model version, and alias state when comparing results.

Changes to data, features, class mapping, estimators, metrics, promotion policy, confidence
semantics, or intended use require a model-card update. See [CONTRIBUTING.md](CONTRIBUTING.md) for
the review checklist and [ROADMAP.md](ROADMAP.md) for planned evaluation work.
