"""Fail CI when a module under api/ exceeds the LOC cap for its layer.

Caps codify the deep-modules invariant. Every ``.py`` under ``api/`` is
matched against the *most specific* prefix in :data:`CAPS`, so each file has
exactly one cap and nested packages cannot escape the guard:

    api/routes/*      HTTP shape + render                      <= 150
    api/services/**   HTTP adapters (incl. nested packages)    <= 250
    api/*             app assembly / DI / job orchestration    <= 600

The third tier exists because ``api/server.py``, ``api/deps.py`` and
``api/jobs.py`` are infrastructure, not request adapters — a 250-line cap
never made sense for them, but leaving them uncapped meant the largest
modules in the HTTP layer were the only ones nobody measured.

Matching is by prefix depth, not glob depth. ``api/services/scenes/_grid.py``
resolves to the ``api/services`` cap (250), not the ``api`` cap (600),
because ``api/services`` is the longer matching prefix.

Bumps require a CHANGELOG entry and reviewer sign-off.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Prefix -> cap. Most specific matching prefix wins; see module docstring.
CAPS = {
    "api": 600,
    "api/routes": 150,
    "api/services": 250,
}

# Files that already meet their cap are NOT exempted — adding them would
# silently disable the guard the moment a future edit pushed them over. Only
# list a file here with a reason and a way out.
#
#   api/routes/preprocess.py (182 > 150) — the Pre-processing tab landed as one
#   route module carrying scene-detect job kickoff, the batched cut-edit
#   endpoints (split/merge), and the filmstrip render. Exempted rather than
#   split mid-feature: the cut-review UX is still moving, and a premature
#   route/service seam would have to be redrawn. Clear it by moving the
#   cut-edit batch handlers behind api/services/preprocess_render.py.
#
#   api/jobs.py (1076 > 600) — pre-existing debt, surfaced (not created) when
#   the api/* tier was added. It carries the in-process job registry, the
#   pipeline step runner, progress/cancellation state, and the SSE event feed
#   in one module. Exempted so the new tier could land without bundling a
#   large refactor. Clear it by lifting the job registry + state machine into
#   src/kuaa/ (they are HTTP-agnostic) and leaving only the SSE adapter here.
EXEMPTIONS: set[str] = {
    "api/routes/preprocess.py",
    "api/jobs.py",
}


def cap_for(rel: str) -> int | None:
    """Return the cap of the most specific configured prefix containing ``rel``.

    Args:
        rel: Repo-relative POSIX path, e.g. ``api/services/scenes/_cards.py``.

    Returns:
        The cap in lines, or ``None`` when no prefix matches.
    """
    best_depth = -1
    best_cap: int | None = None
    for prefix, cap in CAPS.items():
        if rel == prefix or rel.startswith(f"{prefix}/"):
            depth = prefix.count("/")
            if depth > best_depth:
                best_depth, best_cap = depth, cap
    return best_cap


def main() -> int:
    """Report every module over its layer cap; return 1 if any were found."""
    root = Path(__file__).resolve().parent.parent
    violations: list[tuple[str, int, int]] = []
    for py in sorted((root / "api").rglob("*.py")):
        rel = py.relative_to(root).as_posix()
        if rel in EXEMPTIONS:
            continue
        cap = cap_for(rel)
        if cap is None:
            continue
        lines = len(py.read_text().splitlines())
        if lines > cap:
            violations.append((rel, lines, cap))
    if violations:
        for rel, lines, cap in violations:
            print(
                f"LOC BUDGET VIOLATION: {rel} = {lines} lines (cap {cap})",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
