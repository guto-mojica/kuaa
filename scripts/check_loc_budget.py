"""Fail CI when api/services/*.py or api/routes/*.py exceed their LOC caps.

Caps codify the deep-modules invariant: services are HTTP adapters
(<= 250 LOC), routes are HTTP shape + render (<= 150 LOC). Bumps require a
CHANGELOG entry and reviewer sign-off.
"""

from __future__ import annotations

import sys
from pathlib import Path

CAPS = {
    "api/services": 250,
    "api/routes": 150,
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
EXEMPTIONS: set[str] = {
    "api/routes/preprocess.py",
}


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    violations: list[tuple[str, int, int]] = []
    for prefix, cap in CAPS.items():
        for py in (root / prefix).glob("*.py"):
            rel = str(py.relative_to(root))
            if rel in EXEMPTIONS:
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
