#!/usr/bin/env python3
"""Build a deterministic public demo artifact bundle."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import prepare_demo  # noqa: E402

DEFAULT_MANIFEST = REPO_ROOT / "data" / "demo" / "manifest.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "dist" / "demo"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
BUNDLE_ROOTS = ("metadata", "frames", "embeddings")


class BundleBuildError(RuntimeError):
    """Raised when the demo bundle cannot be built safely."""


@dataclass(frozen=True)
class BundleBuildResult:
    """Paths and checksums produced by one bundle build."""

    zip_path: Path
    checksum_path: Path
    manifest_preview_path: Path
    release_snippet_path: Path
    bundle_sha256: str
    file_count: int
    artifact_checksums: dict[str, str]


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _runtime_root(manifest: dict[str, Any], override: str | Path | None) -> Path:
    if override is not None:
        return prepare_demo.project_path(override)
    return prepare_demo.runtime_root_from_manifest(manifest)


def _bundle_filename(manifest: dict[str, Any], override: str | None) -> str:
    if override:
        return Path(override).name
    configured = manifest.get("artifact_bundle", {}).get("filename")
    if configured:
        return Path(str(configured)).name
    return "cinemateca-demo.zip"


def _validate_runtime(runtime_root: Path, manifest: dict[str, Any]) -> prepare_demo.CheckResult:
    result = prepare_demo.check_demo(runtime_root, manifest)
    if result.ok:
        return result

    details = "\n".join(f"- {error}" for error in result.errors)
    raise BundleBuildError(f"Demo runtime validation failed:\n{details}")


def _iter_bundle_files(runtime_root: Path, *, include_raw: bool) -> list[Path]:
    roots = [*BUNDLE_ROOTS]
    if include_raw:
        roots.append("raw")

    files: list[Path] = []
    for root_name in roots:
        root = runtime_root / root_name
        if not root.exists():
            continue
        files.extend(path for path in root.rglob("*") if path.is_file())
    return sorted(files, key=lambda path: path.relative_to(runtime_root).as_posix())


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _write_zip(zip_path: Path, files: list[Path], runtime_root: Path) -> None:
    tmp_path = zip_path.with_suffix(zip_path.suffix + ".part")
    tmp_path.unlink(missing_ok=True)
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            rel = _relative(path, runtime_root)
            info = zipfile.ZipInfo(rel, ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with open(path, "rb") as f:
                archive.writestr(info, f.read())
    tmp_path.replace(zip_path)


def _artifact_checksums(files: list[Path], runtime_root: Path) -> dict[str, str]:
    return {
        _relative(path, runtime_root): prepare_demo.sha256_file(path)
        for path in files
    }


def _manifest_preview(
    manifest: dict[str, Any],
    *,
    zip_path: Path,
    bundle_sha256: str,
    artifact_checksums: dict[str, str],
) -> dict[str, Any]:
    preview = copy.deepcopy(manifest)
    bundle = preview.setdefault("artifact_bundle", {})
    bundle["filename"] = zip_path.name
    bundle["sha256"] = bundle_sha256
    preview["checksums"] = artifact_checksums
    return preview


def _release_snippet(
    *,
    zip_path: Path,
    bundle_sha256: str,
    manifest_preview: dict[str, Any],
    file_count: int,
) -> str:
    bundle = manifest_preview.get("artifact_bundle", {})
    source = manifest_preview.get("source", {})
    return "\n".join(
        [
            "# Demo bundle release snippet",
            "",
            f"- Bundle: `{zip_path.name}`",
            f"- SHA-256: `{bundle_sha256}`",
            f"- Packaged files: `{file_count}`",
            f"- Runtime root: `{bundle.get('root', 'data/demo/runtime')}`",
            f"- Source: {source.get('title', 'public demo source')} "
            f"({source.get('year', 'year unknown')})",
            "",
            "Manifest fields to copy after upload:",
            "",
            "```json",
            json.dumps(
                {
                    "artifact_bundle": bundle,
                    "checksums": manifest_preview.get("checksums", {}),
                },
                indent=2,
                ensure_ascii=False,
            ),
            "```",
            "",
            "Post-upload verification:",
            "",
            "```bash",
            "uv run python scripts/prepare_demo.py --download",
            "uv run python scripts/prepare_demo.py --check",
            "```",
            "",
        ]
    )


def build_demo_bundle(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    runtime_root: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    filename: str | None = None,
    include_raw: bool = True,
    update_manifest: bool = False,
) -> BundleBuildResult:
    """Validate the demo runtime and build release artifacts."""
    manifest_path = Path(manifest_path)
    manifest = prepare_demo.load_manifest(manifest_path)
    root = _runtime_root(manifest, runtime_root)
    _validate_runtime(root, manifest)

    files = _iter_bundle_files(root, include_raw=include_raw)
    if not files:
        raise BundleBuildError(f"No bundle files found under {root}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / _bundle_filename(manifest, filename)
    checksum_path = zip_path.with_name(zip_path.name + ".sha256")
    manifest_preview_path = out / "manifest.preview.json"
    release_snippet_path = out / "release-snippet.md"

    _write_zip(zip_path, files, root)
    bundle_sha = prepare_demo.sha256_file(zip_path)
    artifact_checksums = _artifact_checksums(files, root)
    preview = _manifest_preview(
        manifest,
        zip_path=zip_path,
        bundle_sha256=bundle_sha,
        artifact_checksums=artifact_checksums,
    )

    checksum_path.write_text(f"{bundle_sha}  {zip_path.name}\n", encoding="utf-8")
    _json_dump(manifest_preview_path, preview)
    release_snippet_path.write_text(
        _release_snippet(
            zip_path=zip_path,
            bundle_sha256=bundle_sha,
            manifest_preview=preview,
            file_count=len(files),
        ),
        encoding="utf-8",
    )
    if update_manifest:
        _json_dump(manifest_path, preview)

    return BundleBuildResult(
        zip_path=zip_path,
        checksum_path=checksum_path,
        manifest_preview_path=manifest_preview_path,
        release_snippet_path=release_snippet_path,
        bundle_sha256=bundle_sha,
        file_count=len(files),
        artifact_checksums=artifact_checksums,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--runtime-root", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--filename", default=None)
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Exclude runtime raw/ files from the release ZIP and checksums.",
    )
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="Overwrite the manifest with the generated preview after review.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable output.",
    )
    return parser.parse_args(argv)


def _result_payload(result: BundleBuildResult) -> dict[str, Any]:
    return {
        "zip_path": str(result.zip_path),
        "checksum_path": str(result.checksum_path),
        "manifest_preview_path": str(result.manifest_preview_path),
        "release_snippet_path": str(result.release_snippet_path),
        "bundle_sha256": result.bundle_sha256,
        "file_count": result.file_count,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = build_demo_bundle(
            manifest_path=args.manifest,
            runtime_root=args.runtime_root,
            output_dir=args.output_dir,
            filename=args.filename,
            include_raw=not args.no_raw,
            update_manifest=args.update_manifest,
        )
    except (prepare_demo.DemoError, BundleBuildError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(_result_payload(result), indent=2, sort_keys=True))
    else:
        print("Demo bundle built.")
        print(f"ZIP: {result.zip_path}")
        print(f"SHA-256: {result.bundle_sha256}")
        print(f"Checksum file: {result.checksum_path}")
        print(f"Manifest preview: {result.manifest_preview_path}")
        print(f"Release snippet: {result.release_snippet_path}")
        print(f"Packaged files: {result.file_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
