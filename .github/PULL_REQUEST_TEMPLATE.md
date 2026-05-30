## What & why

<!-- One paragraph. Link the spec item ID(s) (e.g. T3, C1, WS-5) if applicable. -->

## Checklist

- [ ] `uv run ruff check . && uv run black --check .`
- [ ] `uv run mypy src api` is clean (zero errors)
- [ ] `uv run pytest -m "not e2e" -q` passes (coverage floor met)
- [ ] `uv run python scripts/check_loc_budget.py` + `uv run lint-imports` pass
- [ ] Snapshots: no unintended diffs; intended changes are explained above
- [ ] `CHANGELOG.md` updated if user-visible
- [ ] Docs updated (`DESIGN_SYSTEM.md` / API docs / etc.) if applicable

## Behavior / artefact impact

<!-- Does this change observable behavior or generated artefacts? If yes,
     which snapshots/artefacts were regenerated and why (spec §12 budget)? -->
