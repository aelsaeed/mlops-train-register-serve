# Contributing

Thanks for contributing to this project.

## Development setup
1. Create and activate a Python 3.11 virtual environment.
2. Install dependencies:
   ```bash
   make setup
   ```
3. Install git hooks (included in `make setup`):
   ```bash
   pre-commit install
   ```

## Local quality checks
Run before opening a PR:

```bash
make lint
make typecheck
make test
make demo
```

## Pull requests
- Use the pull request template checklist.
- Keep PRs small and scoped.
- Update docs/changelog when behavior changes.
