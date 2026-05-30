"""Fail CI when api/services/*.py or api/routes/*.py exceed their LOC caps.

Caps codify the P1 deep-modules refactor invariant: services are HTTP
adapters (<= 250 LOC), routes are HTTP shape + render (<= 150 LOC). Bumps
require a CHANGELOG entry and reviewer sign-off.
"""

from __future__ import annotations

import sys
from pathlib import Path

CAPS = {
    "api/services": 250,
    "api/routes": 150,
}

# Exemptions during the migration. Each entry names a file currently
# over its cap; the entry stays until the cap-violating commit is
# refactored away (a "P*" phase task). Files that already meet their
# cap are NOT exempted — adding them would silently disable the guard
# the moment a future edit pushed them over.
EXEMPTIONS: set[str] = {
    # P1 will remove these as services slim down.
    "api/services/about_service.py",
    "api/services/processing_service.py",
    "api/routes/scenes.py",
    "api/routes/processing.py",
    "api/routes/library.py",
    "api/routes/annotate.py",
    "api/routes/eval.py",
    # Grew past cap during getting_ready_to_launch(pg) merge (audio search,
    # fusion, rimas MMR, scene list additions). Deferred extraction.
    "api/services/catalog.py",
    "api/routes/search.py",
    "api/routes/rimas.py",
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
