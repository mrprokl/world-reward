# Contributing to World Reward

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Development Workflow

1. Create a branch from `main`.
2. Implement your change with tests.
3. Run the quality gate locally:

```bash
ruff check .
mypy
pytest
```

4. Open a pull request with:
   - Problem statement
   - Change summary
   - Test evidence
   - Risk/rollback notes (if behavior changes)

## Code Style

- Keep modules typed and focused.
- Favor explicit error handling and deterministic behavior.
- Add tests for all non-trivial logic changes.

## Domain Config Contributions

- Add new domain files in `configs/`.
- Keep schema aligned with `DomainConfig`.
- Ensure each category has realistic, physically grounded examples.

## Reporting Bugs

Open an issue with:
- Reproduction steps
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, install method)
