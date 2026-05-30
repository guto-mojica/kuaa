# Cinemateca AI — thin uv task runner. NOT a container; every recipe is a
# one-line `uv run …`. Install just: https://github.com/casey/just
# (Fallback without just: copy any recipe body and run it directly.)

set shell := ["bash", "-uc"]

# Show all recipes.
default:
    @just --list

# Install everything (full ML extra + dev tooling) + pre-commit hooks.
setup:
    uv sync --extra full --group dev
    uv run pre-commit install

# Fast, model-free tests (the default CI gate).
smoke:
    uv run pytest -m smoke -q

# Full suite minus browser tests (coverage floor enforced via pyproject).
test:
    uv run pytest -m "not e2e" -q

# Browser a11y + UI render smoke (needs `uv run playwright install chromium`).
e2e:
    uv run pytest tests/e2e -m e2e -q

# Coverage report to the terminal.
cov:
    uv run pytest -m "not e2e" -q --cov-report=term-missing

# Lint + format check.
lint:
    uv run ruff check .
    uv run black --check .

# Auto-fix lint + format.
fmt:
    uv run ruff check --fix .
    uv run black .

# Static types (blocking gate — must be zero).
type:
    uv run mypy src api

# Security: SAST + dependency CVEs (tools fetched ephemerally via uvx — not project deps).
sec:
    uvx bandit -c pyproject.toml -r src api -ll
    uvx pip-audit --progress-spinner off

# Architecture guards: LOC budget + layer contracts.
guards:
    uv run python scripts/check_loc_budget.py
    uv run lint-imports

# The whole local gate set, fast→slow.
check: lint type smoke guards

# Run the app (FastAPI + HTMX) on 127.0.0.1:8501.
serve:
    uv run cinemateca serve

# Retrieval eval (clip/bm25/hybrid) on the default config.
eval:
    uv run python scripts/run_eval.py

# Latency benchmark on the demo film.
bench:
    uv run python scripts/bench_retrieval.py

# Build sdist + wheel.
build:
    uv build

# Clean-checkout reproducible-run verification (the no-Docker gate).
# Script added in T10 (scripts/verify_fresh_run.sh).
verify:
    bash scripts/verify_fresh_run.sh
