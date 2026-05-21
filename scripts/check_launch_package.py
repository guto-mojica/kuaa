#!/usr/bin/env python3
"""Validate the M5 public launch documentation package."""

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
    """Required launch document structure."""

    path: str
    headings: tuple[str, ...]
    required_links: tuple[str, ...] = ()


@dataclass
class LaunchCheckResult:
    """Result of a launch-package validation run."""

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
            "RELEASE_VERIFICATION.md",
        ),
    ),
    DocRequirement(
        path="docs/LAUNCH_PLAN.md",
        headings=(
            "# Launch plan",
            "## Launch sequence",
            "## Asset map",
            "## Post 1: origin and problem",
            "## Post 2: demo",
            "## Post 3: evaluation",
            "## Post 4: domain adaptation",
            "## Post 5: architecture and production signals",
            "## Final launch checklist",
        ),
        required_links=(
            "CASE_STUDY.md",
            "DEMO_VIDEO_SCRIPT.md",
            "RELEASE_VERIFICATION.md",
        ),
    ),
    DocRequirement(
        path="docs/DEMO_VIDEO_SCRIPT.md",
        headings=(
            "# Demo video scripts",
            "## Preflight",
            "## Two-minute demo",
            "## Eight-to-ten-minute technical walkthrough",
            "## Recording checklist",
        ),
        required_links=(
            "CASE_STUDY.md",
            "DEMO_DATA.md",
            "EVALUATION.md",
            "RELEASE_VERIFICATION.md",
        ),
    ),
    DocRequirement(
        path="docs/RELEASE_NOTES_DRAFT.md",
        headings=(
            "# Release notes draft: public demo launch",
            "## Demo quickstart",
            "## Demo artifact",
            "## Evaluation",
            "## Exports",
            "## Run manifest",
            "## Verification",
            "## Known limits",
        ),
        required_links=(
            "DEMO_DATA.md",
            "RELEASE_VERIFICATION.md",
        ),
    ),
    DocRequirement(
        path="docs/RESUME_BULLETS.md",
        headings=(
            "# Resume and hiring copy",
            "## One-line project summary",
            "## Resume bullets: applied ML engineer",
            "## Resume bullets: backend/platform engineer",
            "## Resume bullets: product-minded AI engineer",
            "## LinkedIn featured project blurb",
            "## Interview talking points",
            "## Claims to avoid until final release verification",
        ),
        required_links=(
            "CASE_STUDY.md",
            "ARCHITECTURE.md",
            "EVALUATION.md",
            "DOMAIN_PACKS.md",
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
        result.errors.append(f"Missing required launch doc: {path}")
    except OSError as exc:
        result.errors.append(f"Could not read launch doc {path}: {exc}")
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
    """Check required M5 launch docs, sections, links, and placeholder tokens."""
    result = LaunchCheckResult()
    root_path = Path(root)
    for requirement in requirements:
        _check_doc(root_path, requirement, result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate M5 launch-package docs.")
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
        print(f"Launch package check passed ({len(result.checked_docs)} docs).")
    else:
        print("Launch package check failed:", file=sys.stderr)
        for error in result.errors:
            print(f"- {error}", file=sys.stderr)

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
