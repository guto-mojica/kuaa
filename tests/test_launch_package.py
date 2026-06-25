from __future__ import annotations

from pathlib import Path

import pytest

from scripts import check_launch_package

_ALL_LAUNCH_DOCS_PRESENT = all(
    (check_launch_package.REPO_ROOT / req.path).exists()
    for req in check_launch_package.DEFAULT_REQUIREMENTS
)


def _requirement() -> check_launch_package.DocRequirement:
    return check_launch_package.DocRequirement(
        path="docs/LAUNCH.md",
        headings=("# Launch", "## Evidence"),
        required_links=("CASE_STUDY.md",),
    )


def _write_launch_doc(root: Path, text: str) -> None:
    path = root / "docs" / "LAUNCH.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.mark.skipif(
    not _ALL_LAUNCH_DOCS_PRESENT,
    reason="local-only launch docs not present — run this test locally before launch",
)
def test_current_launch_package_passes():
    result = check_launch_package.check_launch_package()

    assert result.ok, result.errors
    assert "docs/CASE_STUDY.md" in result.checked_docs
    assert "docs/LAUNCH_PLAN.md" in result.checked_docs


def test_launch_check_reports_missing_doc(tmp_path):
    result = check_launch_package.check_launch_package(
        tmp_path,
        requirements=(_requirement(),),
    )

    assert not result.ok
    assert any("Missing required launch doc" in error for error in result.errors)


def test_launch_check_reports_missing_heading(tmp_path):
    _write_launch_doc(
        tmp_path,
        """# Launch

See CASE_STUDY.md.
""",
    )

    result = check_launch_package.check_launch_package(
        tmp_path,
        requirements=(_requirement(),),
    )

    assert not result.ok
    assert any("missing heading '## Evidence'" in error for error in result.errors)


def test_launch_check_reports_missing_link(tmp_path):
    _write_launch_doc(
        tmp_path,
        """# Launch

## Evidence

No link yet.
""",
    )

    result = check_launch_package.check_launch_package(
        tmp_path,
        requirements=(_requirement(),),
    )

    assert not result.ok
    assert any("missing link/reference 'CASE_STUDY.md'" in error for error in result.errors)


def test_launch_check_reports_placeholder_token(tmp_path):
    _write_launch_doc(
        tmp_path,
        """# Launch

## Evidence

TODO: add launch copy.
See CASE_STUDY.md.
""",
    )

    result = check_launch_package.check_launch_package(
        tmp_path,
        requirements=(_requirement(),),
    )

    assert not result.ok
    assert any("unresolved placeholder token" in error for error in result.errors)
