## What & why

<!-- One paragraph: the change and the reason for it. -->

## Checklist

- [ ] `uv run ruff check . && uv run ruff format --check .`
- [ ] `uv run mypy src api` is clean (zero errors)
- [ ] `uv run pytest -m "not e2e" -q` passes (coverage floor met)
- [ ] `uv run python scripts/check_loc_budget.py` + `uv run lint-imports` pass
- [ ] `uv run python scripts/check_launch_package.py` passes if a gated doc changed
- [ ] Snapshots: no unintended diffs; intended changes are explained above
- [ ] `CHANGELOG.md` updated if user-visible
- [ ] Docs updated (`docs/ARCHITECTURE.md`, `docs/API.md`, etc.) if applicable

## Behavior / artefact impact

<!-- Does this change observable behavior or generated artefacts? If yes, which
     snapshots/artefacts were regenerated, and why? Changes to the AI pipeline
     (model swap, prompt change, dependency bump) can invalidate already-generated
     artefacts — call that out explicitly. -->
