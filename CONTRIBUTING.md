# Contributing

Contributions should keep the project easy to evaluate locally and should make model behavior,
lineage, and delivery decisions observable.

## Development setup

Prerequisites:

- Python 3.11 or 3.12
- `make`
- Git
- Docker with Compose for container-backed integration work

Create an isolated environment and install the package with its development dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
make setup
```

`make setup` performs an editable install constrained by `constraints.txt` and installs the
repository's pre-commit hooks.

## Validate a change

Use `make test` while iterating. Before opening a pull request, run the full local gate:

```bash
make check
make test
make demo
make build
```

`make check` runs the formatting check, Ruff, mypy, and tests with the configured coverage
threshold. `make build` verifies the wheel and source distribution.

For changes to containers or service wiring, also validate the Compose configuration and the
full lifecycle:

```bash
docker compose config
bash scripts/demo.sh --full
```

Use `docker-compose config` on a legacy Compose installation; the lifecycle script detects both
command forms.

Do not commit files generated under `artifacts/`, MLflow databases, caches, or local virtual
environments.

## Change expectations

- **Training or evaluation:** add tests for deterministic behavior and update the model card when
  data, features, metrics, intended use, or limitations change.
- **Registry or promotion:** test both passing and failing gates. A failure must leave the current
  `champion` unchanged.
- **API:** preserve the OpenAPI contract or document the breaking change; cover validation,
  batch behavior, readiness, and metrics.
- **Observability:** use stable metric names and labels, avoid high-cardinality values, and do not
  expose raw feature payloads.
- **Dependencies or containers:** pin deliberately, explain compatibility changes, and confirm a
  clean build.
- **Documentation:** keep commands copy/paste-ready and never publish unverified benchmark or
  runtime claims.

## Pull requests

Use a short branch name consistent with the repository, such as `feature/promotion-policy`,
`fix/readiness-check`, or `docs/model-card`.

A pull request should:

1. Explain the problem and the reason for the chosen design.
2. Describe behavior and compatibility changes.
3. Include tests or explain why tests are not applicable.
4. Report the exact validation commands that were run.
5. Update `README.md`, `MODEL_CARD.md`, `CHANGELOG.md`, or `ROADMAP.md` when their claims change.
6. Include screenshots or example requests for user-visible API or MLflow changes.
7. Call out rollout, rollback, data, security, and model-quality risks where relevant.

Keep pull requests focused enough that a reviewer can connect the design decision, implementation,
and evidence. Use the repository's pull request template and wait for required CI checks to pass
before merging.

## Reporting problems and proposing work

Use the GitHub issue templates for reproducible defects and feature proposals. Include the Python
version, operating system, command, relevant configuration, and sanitized logs. Never include
credentials, tokens, private data, or unredacted prediction payloads.

For larger changes, open an issue first so the interface, evaluation criteria, and scope can be
agreed before implementation.
