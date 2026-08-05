#!/usr/bin/env python3
"""Validate the public documentation package.

Gates the reader-facing docs a visitor is pointed at from ``README.md``: each
must exist, keep its load-bearing headings, still link the docs it promises,
and carry no unresolved placeholder token. Run by the ``docs`` job in
``.github/workflows/ci.yml``.

Scope note: this checks the docs that ship in the public tree. Internal
planning material (agent specs, conversation logs, career copy) was removed
from the tracked tree in ``chore(repo): public launch cleanup`` and is
gitignored; it is deliberately not gated here.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DocRequirement:
    """Required public-document structure."""

    path: str
    headings: tuple[str, ...]
    required_links: tuple[str, ...] = ()


@dataclass
class LaunchCheckResult:
    """Result of a public-docs validation run."""

    checked_docs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "checked_docs": self.checked_docs,
            "errors": self.errors,
        }


DEFAULT_REQUIREMENTS: tuple[DocRequirement, ...] = (
    DocRequirement(
        path="docs/CASE_STUDY.md",
        headings=(
            "# Case study: offline multimodal search for archival video",
            "## Problem",
            "## Constraints",
            "## What I built",
            "## Evaluation",
            "## Domain adaptation",
            "## Production signals",
            "## Limitations",
            "## Next steps",
        ),
        required_links=(
            "ARCHITECTURE.md",
            "EVALUATION.md",
            "DOMAIN_PACKS.md",
            "OPERATIONS.md",
        ),
    ),
    DocRequirement(
        path="docs/PROJECT_BRIEF.md",
        headings=(
            "# Project brief",
            "## Summary",
            "## Problem",
            "## Current capabilities",
            "## What this project is not",
            "## Next proof points",
        ),
    ),
    DocRequirement(
        path="docs/DEMO.md",
        headings=(
            "# Reproducible Demo",
            "## Five-Minute Quickstart",
            "## Runtime Layout",
            "## Full Processing Path",
            "## Two-minute walkthrough",
        ),
        required_links=(
            "DEMO_DATA.md",
            "EVALUATION.md",
        ),
    ),
    DocRequirement(
        path="docs/DEMO_DATA.md",
        headings=(
            "# Demo Data",
            "## Primary Demo Source",
            "## Artifact Policy",
            "## Updating The Demo Bundle",
        ),
    ),
)

PLACEHOLDER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("TODO", re.compile(r"\bTODO\b", re.IGNORECASE)),
    ("TBD", re.compile(r"\bTBD\b", re.IGNORECASE)),
    ("FIXME", re.compile(r"\bFIXME\b", re.IGNORECASE)),
    ("REPLACE_ME", re.compile(r"\bREPLACE_ME\b", re.IGNORECASE)),
    ("YOUR_*", re.compile(r"\bYOUR_[A-Z0-9_]+\b")),
    ("double braces", re.compile(r"\{\{[^}]+\}\}")),
    ("double brackets", re.compile(r"\[\[[^\]]+\]\]")),
    ("insert tag", re.compile(r"<(?:insert|replace|todo|tbd)[^>]*>", re.IGNORECASE)),
    ("lorem ipsum", re.compile(r"lorem ipsum", re.IGNORECASE)),
)


def _read_text(path: Path, result: LaunchCheckResult) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        result.errors.append(f"Missing required public doc: {path}")
    except OSError as exc:
        result.errors.append(f"Could not read public doc {path}: {exc}")
    return None


def _line_number(text: str, pattern: re.Pattern[str]) -> int:
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern.search(line):
            return line_no
    return 0


def _check_doc(root: Path, requirement: DocRequirement, result: LaunchCheckResult) -> None:
    path = root / requirement.path
    text = _read_text(path, result)
    if text is None:
        return

    result.checked_docs.append(requirement.path)
    headings = {line.strip() for line in text.splitlines() if line.startswith("#")}
    for heading in requirement.headings:
        if heading not in headings:
            result.errors.append(f"{requirement.path}: missing heading {heading!r}")

    for link in requirement.required_links:
        if link not in text:
            result.errors.append(f"{requirement.path}: missing link/reference {link!r}")

    for label, pattern in PLACEHOLDER_PATTERNS:
        if pattern.search(text):
            result.errors.append(
                f"{requirement.path}:{_line_number(text, pattern)}: "
                f"unresolved placeholder token ({label})"
            )


def check_launch_package(
    root: str | Path = REPO_ROOT,
    requirements: Iterable[DocRequirement] = DEFAULT_REQUIREMENTS,
) -> LaunchCheckResult:
    """Check required public docs, sections, links, and placeholder tokens."""
    result = LaunchCheckResult()
    root_path = Path(root)
    for requirement in requirements:
        _check_doc(root_path, requirement, result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the public docs package.")
    parser.add_argument(
        "--root",
        default=str(REPO_ROOT),
        help="Repository root to validate. Defaults to this checkout.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable validation output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = check_launch_package(args.root)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    elif result.ok:
        print(f"Public docs check passed ({len(result.checked_docs)} docs).")
    else:
        print("Public docs check failed:", file=sys.stderr)
        for error in result.errors:
            print(f"- {error}", file=sys.stderr)

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
