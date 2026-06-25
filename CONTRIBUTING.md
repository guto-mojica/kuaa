# Contributing to KUAA AI

Thanks for contributing. This project is `uv`-only — **no Docker, no npm/node,
no SPA build step** (see `CLAUDE.md` "Canonical stack"). All commands below
assume `uv` is installed (<https://docs.astral.sh/uv/>).

---

## Setup

```bash
uv venv                          # creates .venv from .python-version (3.11)
uv sync --extra full --group dev # full ML extra + dev tooling
uv run pre-commit install        # ruff + black + bandit on every commit
```

Run the app:

```bash
uv run kuaa serve          # http://127.0.0.1:8501 (--reload by default)
```

---

## The quality gates

All gates run in CI. Run them locally before pushing.

| Gate | Command | Blocking? |
|---|---|---|
| Lint | `uv run ruff check . && uv run black --check .` | yes |
| Types | `uv run mypy src api` | yes (zero errors) |
| Smoke tests | `uv run pytest -m smoke -q` | yes |
| Full tests + coverage | `uv run pytest -m "not e2e" -q` | yes (≥75% coverage floor) |
| E2E (a11y + UI) | `uv run pytest tests/e2e -m e2e -q` | yes |
| Security | `uv run bandit -c pyproject.toml -r src api -ll && uv run pip-audit` | yes |
| LOC budget | `uv run python scripts/check_loc_budget.py` | yes |
| Layer rules | `uv run lint-imports` | yes |
| Build | `uv build` | yes |
| Fresh-run | `scripts/verify_fresh_run.sh` | release gate |

When available, `just check` (see `justfile`) runs lint + types + smoke + coverage
in one shot.

---

## Commit convention

`type(scope): short description`

Types: `feat`, `fix`, `refactor`, `docs`, `chore`, `test`.

Example: `feat(api): add /api/library/tree endpoint with film grouping`.

---

## Updating golden snapshots

Refactors are gated by golden snapshots so behavior change is explicit.
Snapshots live in `tests/fixtures/snapshots/`. When a change *intentionally*
alters output, regenerate the snapshot in the **same** commit and explain
why in the commit message:

```bash
# Unified helper (F4) — preferred:
UPDATE_SNAPSHOTS=1 uv run pytest tests/<the_snapshot_test>.py

# Per-feature flag (legacy, still in use for some tests):
UPDATE_P1_SNAPSHOT=1 uv run pytest tests/test_p1_search_snapshot.py
```

Never update a snapshot to "make CI green" without understanding the diff.
A behavior-changing item (e.g. reranker default-ON, tokenizer swap) must say
so explicitly and refresh artefacts per the spec §12 regeneration budget.

---

## Test markers

| Marker | Meaning |
|---|---|
| `smoke` | Fast, model-free; the default CI gate. Unmarked tests run here too. |
| `heavy` | Imports a model wheel; runs only in the full-extra matrix job. |
| `acceptance` | Needs real Jeca Tatu artefacts on disk; skipped when absent. |
| `e2e` | Playwright browser tests (a11y + UI render smoke). |

Mark new tests accordingly. Tests that import `torch` or model weights must
be marked `heavy` or `acceptance`.

---

## Layering rules

`kuaa.*` packages must not import from `api/*`. This is enforced by
`import-linter` (`.importlinter`). The public surface of each package lives
in its `__init__.py`; everything else is an implementation detail.

Per-module LOC budgets are enforced by `scripts/check_loc_budget.py`.
Both run in CI (`.github/workflows/refactor-guards.yml`).

See `docs/` for architecture and design decisions:
- `docs/DESIGN_SYSTEM.md` — visual source of truth (colors, typography, components).
- `CLAUDE.md` — operational briefing, vocabulary, and coding conventions.

When a design decision lands in code, update `docs/DESIGN_SYSTEM.md` if it
establishes a new pattern.

---

## CI-gated launch docs

`scripts/check_launch_package.py` (CI) requires these five docs at their exact
paths, with their exact headings and required link substrings, and **fails on any
placeholder token** (`TODO`/`TBD`/`FIXME`/`REPLACE_ME`/`YOUR_*`/`{{…}}`/`[[…]]`/`lorem ipsum`):

- `docs/CASE_STUDY.md`, `docs/LAUNCH_PLAN.md`, `docs/DEMO_VIDEO_SCRIPT.md`, `docs/RELEASE_NOTES_DRAFT.md`, `docs/RESUME_BULLETS.md`

`docs/DEMO_DATA.md` is a required link substring in two of them, so it cannot be
moved or deleted either. After editing any gated doc, run
`uv run python scripts/check_launch_package.py` and confirm it passes. All other
docs are ungated and may carry draft markers (e.g. `docs/launch/BLOG_OUTLINE.md`).

---

## What Claude Code will not do without explicit request

See `CLAUDE.md` for the full list. In particular: no force-push, no history
rewrite, no unilateral merge-conflict resolution.
