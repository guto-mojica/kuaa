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

# Exemptions during the migration. Each entry is a `(path, until_commit)`
# tuple, NOT a permanent allowance — remove once the migration task
# specified in the spec lands.
EXEMPTIONS: set[str] = {
    # P1 will remove these as services slim down.
    "api/services/search.py",
    "api/services/scenes_service.py",
    "api/services/annotations.py",
    "api/services/eval_service.py",
    "api/services/rhymes_service.py",
    "api/services/catalog.py",
    "api/services/about_service.py",
    "api/services/chrome_service.py",
    "api/services/processing_service.py",
    "api/services/film_context.py",
    "api/services/palette_service.py",
    "api/routes/search.py",
    "api/routes/scenes.py",
    "api/routes/processing.py",
    "api/routes/library.py",
    "api/routes/annotate.py",
    "api/routes/about.py",
    "api/routes/eval.py",
}


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    violations: list[tuple[str, int, int]] = []
    for prefix, cap in CAPS.items():
        for py in (root / prefix).glob("*.py"):
            rel = str(py.relative_to(root))
            if rel in EXEMPTIONS:
                continue
            lines = py.read_text().count("\n")
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
