#!/usr/bin/env python3
"""Prepare or validate the M1 public demo artifact bundle.

The populated demo is intentionally distributed as a release ZIP instead of
tracked binary data. This script downloads that bundle when requested, validates
the local layout, and prints the exact command for launching the demo config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "demo" / "manifest.json"
DEFAULT_RUNTIME_ROOT = REPO_ROOT / "data" / "demo" / "runtime"

DEFAULT_METADATA_FILES = (
    "keyframes_metadata.json",
    "scene_descriptions.json",
    "scene_tags.json",
    "visual_analysis.json",
)
EMBEDDINGS_FILENAME = "keyframe_embeddings.npy"
MAPPING_FILENAME = "index_mapping.json"
VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v", ".webm", ".mkv")
KEYFRAME_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


class DemoError(RuntimeError):
    """Raised for a clear user-facing demo preparation failure."""


@dataclass
class CheckResult:
    """Structured result for demo artifact validation."""

    runtime_root: Path
    scene_count: int = 0
    keyframe_count: int = 0
    embedding_count: int | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise DemoError(f"Demo manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise DemoError(f"Demo manifest is invalid JSON: {manifest_path}: {exc}") from exc


def project_path(value: str | Path, *, base: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def runtime_root_from_manifest(manifest: dict[str, Any]) -> Path:
    configured = manifest.get("artifact_bundle", {}).get("root")
    if not configured:
        return DEFAULT_RUNTIME_ROOT
    return project_path(configured)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path, result: CheckResult) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        result.errors.append(f"Missing required file: {path}")
    except json.JSONDecodeError as exc:
        result.errors.append(f"Invalid JSON: {path}: {exc}")
    except OSError as exc:
        result.errors.append(f"Could not read {path}: {exc}")
    return None


def _is_inside(path: Path, root: Path) -> bool:
    try:
        resolved = path.resolve()
        resolved_root = root.resolve()
    except OSError:
        return False
    return resolved == resolved_root or resolved_root in resolved.parents


def _artifact_candidates(raw_path: str | Path, runtime_root: Path) -> list[Path]:
    path = Path(raw_path)
    if path.is_absolute():
        return [path]

    candidates = [REPO_ROOT / path, runtime_root / path]
    parts = path.parts
    if parts and parts[0] in {"frames", "metadata", "embeddings", "raw"}:
        candidates.insert(0, runtime_root / path)
    return list(dict.fromkeys(candidates))


def _validate_keyframe_paths(
    keyframes_meta: list[Any], runtime_root: Path, result: CheckResult
) -> None:
    for idx, record in enumerate(keyframes_meta):
        if not isinstance(record, dict):
            result.errors.append(f"keyframes_metadata[{idx}] is not an object")
            continue
        raw = record.get("filepath")
        if not raw:
            result.errors.append(f"keyframes_metadata[{idx}] has no filepath")
            continue

        existing = [p for p in _artifact_candidates(raw, runtime_root) if p.exists()]
        if not existing:
            result.errors.append(f"Keyframe file listed in metadata is missing: {raw}")
            continue
        if not any(_is_inside(p, runtime_root) for p in existing):
            result.errors.append(
                f"Keyframe filepath must resolve inside the demo runtime root: {raw}"
            )


def _validate_embeddings(root: Path, result: CheckResult) -> None:
    emb_path = root / "embeddings" / EMBEDDINGS_FILENAME
    map_path = root / "embeddings" / MAPPING_FILENAME
    if not emb_path.exists():
        result.errors.append(f"Missing required file: {emb_path}")
        return
    mapping = _load_json(map_path, result)
    if not isinstance(mapping, dict):
        return

    try:
        import numpy as np

        embeddings = np.load(emb_path, mmap_mode="r")
        row_count = int(embeddings.shape[0])
    except Exception as exc:
        result.errors.append(f"Could not load embeddings array {emb_path}: {exc}")
        return

    keyframe_paths = mapping.get("keyframe_paths")
    scene_ids = mapping.get("scene_ids")
    declared = mapping.get("total_vectors")
    if not isinstance(keyframe_paths, list):
        result.errors.append(f"{map_path} must contain list keyframe_paths")
        return
    if not isinstance(scene_ids, list):
        result.errors.append(f"{map_path} must contain list scene_ids")
        return

    result.embedding_count = row_count
    if row_count != len(keyframe_paths):
        result.errors.append(
            "Embedding row count does not match index mapping: "
            f"{row_count} rows vs {len(keyframe_paths)} keyframe_paths"
        )
    if len(scene_ids) != len(keyframe_paths):
        result.errors.append(
            "Index mapping scene_ids count does not match keyframe_paths: "
            f"{len(scene_ids)} vs {len(keyframe_paths)}"
        )
    if declared is not None and declared != len(keyframe_paths):
        result.errors.append(
            "Index mapping total_vectors does not match keyframe_paths: "
            f"{declared} vs {len(keyframe_paths)}"
        )


def _validate_checksums(runtime_root: Path, manifest: dict[str, Any], result: CheckResult) -> None:
    checksums = manifest.get("checksums") or {}
    if not isinstance(checksums, dict):
        result.errors.append("manifest checksums must be an object")
        return

    for rel_path, expected in checksums.items():
        if not expected:
            continue
        path = runtime_root / rel_path
        if not path.exists():
            result.errors.append(f"Checksum target is missing: {path}")
            continue
        actual = sha256_file(path)
        if actual.lower() != str(expected).lower():
            result.errors.append(
                f"Checksum mismatch for {rel_path}: expected {expected}, got {actual}"
            )


def check_demo(
    runtime_root: str | Path,
    manifest: dict[str, Any] | None = None,
) -> CheckResult:
    """Validate the local demo runtime layout without using the network."""
    manifest = manifest or load_manifest()
    root = Path(runtime_root)
    result = CheckResult(runtime_root=root)

    if not root.exists():
        result.errors.append(f"Demo runtime root does not exist: {root}")
        return result

    metadata_dir = root / "metadata"
    frames_dir = root / "frames" / "scenes" / "keyframes_content"
    raw_dir = root / "raw"

    expected = manifest.get("expected") or {}
    required_metadata = expected.get("required_metadata") or list(DEFAULT_METADATA_FILES)
    for filename in required_metadata:
        path = metadata_dir / filename
        if not path.exists():
            result.errors.append(f"Missing required file: {path}")

    keyframes_meta = _load_json(metadata_dir / "keyframes_metadata.json", result)
    if isinstance(keyframes_meta, list):
        result.scene_count = len(keyframes_meta)
        min_keyframes = int(expected.get("min_keyframes") or 1)
        if len(keyframes_meta) < min_keyframes:
            result.errors.append(
                "Not enough scenes in keyframes_metadata.json: "
                f"{len(keyframes_meta)} < {min_keyframes}"
            )
        _validate_keyframe_paths(keyframes_meta, root, result)
    elif keyframes_meta is not None:
        result.errors.append("keyframes_metadata.json must contain a list")

    for filename in required_metadata:
        if filename == "keyframes_metadata.json":
            continue
        _load_json(metadata_dir / filename, result)

    if frames_dir.exists():
        result.keyframe_count = sum(
            1 for p in frames_dir.rglob("*") if p.suffix.lower() in KEYFRAME_SUFFIXES
        )
    else:
        result.errors.append(f"Missing keyframe directory: {frames_dir}")

    if result.scene_count and result.keyframe_count < result.scene_count:
        result.warnings.append(
            "Fewer keyframe image files than scene records: "
            f"{result.keyframe_count} images vs {result.scene_count} scenes"
        )

    if raw_dir.exists():
        videos = [p for p in raw_dir.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES]
        if not videos:
            result.warnings.append(
                "No source video found under raw/. The populated Search, Scenes, "
                "and Annotate demo can still run, but Processing will have no "
                "demo video to select."
            )
    else:
        result.warnings.append(
            "No raw/ directory found. The populated demo can still run from precomputed artifacts."
        )

    _validate_embeddings(root, result)
    _validate_checksums(root, manifest, result)
    return result


def _request(url: str):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "kuaa-imgsearch-demo/0.3"},
    )
    return urllib.request.urlopen(req, timeout=60)


def _download_file(url: str, dest: Path, expected_sha256: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with _request(url) as response, open(tmp, "wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise DemoError(f"Download failed: {url}: {exc}") from exc

    if expected_sha256:
        actual = sha256_file(tmp)
        if actual.lower() != expected_sha256.lower():
            tmp.unlink(missing_ok=True)
            raise DemoError(
                f"Checksum mismatch for {dest.name}: expected {expected_sha256}, got {actual}"
            )
    tmp.replace(dest)


def _find_video_urls(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for child in value.values():
            found.extend(_find_video_urls(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_find_video_urls(child))
    elif isinstance(value, str):
        lower = value.lower().split("?", 1)[0]
        if value.startswith("http") and lower.endswith(VIDEO_SUFFIXES):
            found.append(value)
    return found


def discover_source_video_url(manifest: dict[str, Any]) -> str | None:
    json_url = manifest.get("source", {}).get("loc_item_json_url")
    if not json_url:
        return None
    try:
        with _request(json_url) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None

    urls = _find_video_urls(payload)
    if not urls:
        return None
    return sorted(urls, key=lambda u: (not u.lower().split("?", 1)[0].endswith(".mp4"), u))[0]


def _zip_has_layout(names: list[str]) -> bool:
    roots = {"metadata/", "frames/", "embeddings/", "raw/"}
    return any(any(name.startswith(prefix) for prefix in roots) for name in names)


def _safe_extract(
    archive: zipfile.ZipFile,
    target: Path,
    names: list[str],
    *,
    strip_prefix: str | None = None,
) -> None:
    target = target.resolve()
    for info in archive.infolist():
        name = info.filename
        if name.endswith("/") or name not in names:
            continue
        rel = name
        if strip_prefix:
            rel = rel[len(strip_prefix) :]
        dest = (target / rel).resolve()
        if target != dest and target not in dest.parents:
            raise DemoError(f"Refusing unsafe zip path: {name}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as src, open(dest, "wb") as dst:
            dst.write(src.read())


def extract_bundle(zip_path: Path, runtime_root: Path) -> None:
    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise DemoError(f"Artifact bundle is not a valid ZIP: {zip_path}") from exc

    with archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]
        if any(n.startswith("data/demo/runtime/") for n in names):
            _safe_extract(archive, REPO_ROOT, names)
            return
        if _zip_has_layout(names):
            _safe_extract(archive, runtime_root, names)
            return

        first_parts = {Path(n).parts[0] for n in names if Path(n).parts}
        if len(first_parts) == 1:
            prefix = next(iter(first_parts)) + "/"
            stripped = [n[len(prefix) :] for n in names if n.startswith(prefix)]
            if _zip_has_layout(stripped):
                _safe_extract(archive, runtime_root, names, strip_prefix=prefix)
                return

        raise DemoError(
            "Artifact ZIP does not contain the expected metadata/, frames/, and embeddings/ layout."
        )


def prepare_downloads(
    manifest: dict[str, Any],
    runtime_root: Path,
    *,
    bundle_url: str | None = None,
    source_video_url: str | None = None,
    skip_video: bool = False,
) -> None:
    bundle = manifest.get("artifact_bundle", {})
    url = bundle_url or bundle.get("url")
    if not url:
        raise DemoError("No artifact bundle URL is configured in the manifest.")

    filename = bundle.get("filename") or Path(url).name
    bundle_path = runtime_root.parent / filename
    expected_bundle_sha = bundle.get("sha256")
    if not bundle_path.exists() or (
        expected_bundle_sha and sha256_file(bundle_path).lower() != expected_bundle_sha.lower()
    ):
        print(f"Downloading demo artifact bundle: {url}")
        _download_file(url, bundle_path, expected_bundle_sha)
    else:
        print(f"Using existing artifact bundle: {bundle_path}")
    extract_bundle(bundle_path, runtime_root)

    if skip_video:
        return

    source = manifest.get("source", {})
    raw_dir = runtime_root / "raw"
    video_name = source.get("source_video_filename") or "demo-source.mp4"
    video_path = raw_dir / video_name
    expected_video_sha = source.get("source_video_sha256")
    if video_path.exists() and (
        not expected_video_sha or sha256_file(video_path).lower() == expected_video_sha.lower()
    ):
        print(f"Using existing source video: {video_path}")
        return

    video_url = (
        source_video_url or source.get("source_video_url") or discover_source_video_url(manifest)
    )
    if not video_url:
        item_url = source.get("loc_item_url", "the source item page")
        raise DemoError(
            "Could not discover a direct source-video download URL. "
            f"Download the video manually from {item_url} into {video_path}, "
            "or rerun with --source-video-url."
        )

    print(f"Downloading source video: {video_url}")
    _download_file(video_url, video_path, expected_video_sha)


def print_result(result: CheckResult) -> None:
    print(f"Demo runtime: {result.runtime_root}")
    print(f"Scenes: {result.scene_count}")
    print(f"Keyframes: {result.keyframe_count}")
    if result.embedding_count is not None:
        print(f"Embeddings: {result.embedding_count}")

    for warning in result.warnings:
        print(f"WARNING: {warning}")
    for error in result.errors:
        print(f"ERROR: {error}")

    if result.ok:
        print("Demo artifacts are valid.")
        print("Run: uv run app.py --config config/demo.yaml")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--runtime-root", default=None)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the configured release artifact bundle before checking.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate local artifacts. This is the default when --download is omitted.",
    )
    parser.add_argument("--bundle-url", default=None)
    parser.add_argument("--source-video-url", default=None)
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Do not download the source video; useful when only browsing precomputed artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        runtime_root = (
            project_path(args.runtime_root)
            if args.runtime_root
            else runtime_root_from_manifest(manifest)
        )
        if args.download:
            prepare_downloads(
                manifest,
                runtime_root,
                bundle_url=args.bundle_url,
                source_video_url=args.source_video_url,
                skip_video=args.skip_video,
            )

        result = check_demo(runtime_root, manifest)
        print_result(result)
        return 0 if result.ok else 1
    except DemoError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
