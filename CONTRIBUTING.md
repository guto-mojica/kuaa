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

Most gates run in CI. The table says which workflow enforces each one — the
last three are local/manual and are *not* enforced on a PR.

| Gate | Command | Enforced by |
|---|---|---|
| Lint + format | `uv run ruff check . && uv run ruff format --check .` | `ci.yml` (lint) |
| Public docs | `uv run python scripts/check_launch_package.py` | `ci.yml` (docs) |
| Smoke tests | `uv run pytest tests/ -q -m smoke` | `ci.yml` (test-smoke) |
| Full tests + coverage | `uv run pytest -m "not e2e" -q` | `ci.yml` (test-heavy), ≥75% floor |
| Types | `uv run mypy src api` | `ci.yml` (typecheck), zero errors |
| Security | `uvx bandit -c pyproject.toml -r src api -ll && uvx pip-audit` | `ci.yml` (security) |
| Build | `uv build` | `ci.yml` (build) |
| LOC budget | `uv run python scripts/check_loc_budget.py` | `refactor-guards.yml` |
| Layer rules | `uv run lint-imports` | `refactor-guards.yml` |
| Retrieval latency | `uv run python scripts/bench_retrieval.py --smoke` | `ci.yml` (benchmark), non-blocking |
| E2E (a11y + UI) | `just e2e` | **local only** — Playwright is not installed in CI |
| Fresh-run | `bash scripts/verify_fresh_run.sh` | **local only** — run before a release |

`just check` runs lint + types + docs + smoke + guards in one shot. `just e2e`
needs the optional Playwright group: `uv sync --group e2e && uv run playwright
install chromium`.

**Formatter of record is `ruff format`.** Black is not a project dependency;
running it will reformat against CI. Pre-commit installs the matching
`ruff-format` hook.

---

## Commit convention

`type(scope): short description`

Types: `feat`, `fix`, `refactor`, `docs`, `chore`, `test`.

Example: `feat(api): add /api/library/tree endpoint with film grouping`.

The maintainer is the sole author: **do not add `Co-Authored-By` trailers.**

### Work-item IDs in older text

`CHANGELOG.md`, some docstrings, and some CI comments cite short IDs — `WS-2`,
`M3`, `C5`, `T9`, `P1`, `U5`, `E4`, `Task 13`. These came from internal planning
specs that are no longer in the public tree, so **no public doc defines them**.
Treat them as historical provenance markers: they tell you a change belonged to
some batch of work, and nothing more. Nothing in the repo resolves them, and you
never need to.

Do not add new ones. Describe the change instead — a reader should not need a
missing document to understand a comment.

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
A behavior-changing item (e.g. reranker default-ON, tokenizer swap) must say so
explicitly in the commit message, and must state which generated artefacts it
invalidates. Changes to the AI pipeline — model swap, Moondream prompt edit,
dependency version bump — can invalidate already-generated per-film artefacts;
per `CLAUDE.md`, check with the maintainer before applying one.

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

- `docs/ARCHITECTURE.md` — module layout, artifact contracts, job policy.
- `docs/API.md` — REST/HTMX route surface.
- `CLAUDE.md` — operational briefing, vocabulary, and coding conventions.

There is no separate design-system doc. The visual source of truth is the CSS
itself (`web/static/css/`), and the UI vocabulary is the table in `CLAUDE.md`.

---

## CI-gated public docs

`scripts/check_launch_package.py` runs in CI (the `docs` job in `ci.yml`). It
requires these docs at their exact paths, with their listed headings and
required link substrings, and **fails on any placeholder token**
(`TODO`/`TBD`/`FIXME`/`REPLACE_ME`/`YOUR_*`/`{{…}}`/`[[…]]`/`lorem ipsum`):

- `docs/CASE_STUDY.md`, `docs/PROJECT_BRIEF.md`, `docs/DEMO.md`, `docs/DEMO_DATA.md`

Renaming a heading or dropping a promised link in one of these breaks the build.
After editing a gated doc, run `uv run python scripts/check_launch_package.py`
(or `just docs`) and confirm it passes. Every other doc is ungated and may carry
draft markers.

Internal planning material — agent specs, conversation logs, career copy — was
removed from the tracked tree in `chore(repo): public launch cleanup` and is
gitignored. Do not reintroduce it, and do not add references to it from public
docs or docstrings.

---

## What Claude Code will not do without explicit request

See `CLAUDE.md` for the full list. In particular: no force-push, no history
rewrite, no unilateral merge-conflict resolution.
