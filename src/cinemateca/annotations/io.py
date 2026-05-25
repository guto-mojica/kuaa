"""Annotation I/O — read/write manual_annotations.json + merge with LLM tags."""

from __future__ import annotations

import json
import logging
import os
import stat
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cinemateca.library.context import FilmContext

logger = logging.getLogger(__name__)

FILENAME = "manual_annotations.json"


def atomic_write_json(path: Path, data: object) -> Path:
    """Atomically write ``data`` as JSON to ``path``.

    Serialises to a same-directory temp file, preserves the target's existing
    permissions (or applies the umask default for new files), then calls
    ``os.replace`` which is atomic on POSIX.  A crash mid-write leaves the
    original file intact — the temp is cleaned up on any exception.

    This is the shared primitive underlying both :func:`save` and
    ``api.services.annotations.save_description``.  It is exported so service
    code can delegate without duplicating the logic.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        if path.exists():
            os.chmod(tmp_path, stat.S_IMODE(os.stat(path).st_mode))
        else:
            current = os.umask(0)
            os.umask(current)
            os.chmod(tmp_path, 0o666 & ~current)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return path


def load(metadata_dir: str | Path) -> dict[str, list[str]]:
    """
    Carrega anotações manuais do disco.

    Returns:
        Dict de {scene_id (str): [tags]}. Vazio se o arquivo não existir.
    """
    path = Path(metadata_dir) / FILENAME
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    logger.debug("Anotações manuais carregadas: %d cenas anotadas", len(data))
    return data


def save(metadata_dir: str | Path, annotations: dict[str, list[str]]) -> Path:
    """Persist the annotations dict to ``manual_annotations.json`` atomically.

    Delegates to :func:`atomic_write_json` for the crash-safe write.  The
    on-disk bytes (JSON ``indent=2, ensure_ascii=False``) and file
    permissions are identical to the old plain-rewrite path.

    Args:
        metadata_dir:  Diretório de metadados do projeto.
        annotations:   Dict {scene_id (str): [tags]}.

    Returns:
        Path do arquivo salvo.
    """
    path = Path(metadata_dir) / FILENAME
    atomic_write_json(path, annotations)
    logger.info("✓ Anotações manuais salvas: %s (%d cenas)", path, len(annotations))
    return path


def merge_tag_index(
    llm_tag_index: dict[str, list[str]] | None,
    annotations: dict[str, list[str]],
) -> dict[str, list[str]]:
    """
    Mescla o índice de tags do LLM com as anotações manuais.

    O índice de tags do LLM tem o formato invertido:
        { "tag": ["scene_id1", "scene_id2"] }

    As anotações manuais têm o formato:
        { "scene_id": ["tag1", "tag2"] }

    Returns:
        Índice mesclado no formato { "tag": [scene_ids] }.
    """
    merged: dict[str, list[str]] = {}

    # Copiar tags do LLM
    if llm_tag_index:
        for tag, ids in llm_tag_index.items():
            merged[tag] = list(ids)

    # Adicionar tags manuais
    for scene_id, tags in annotations.items():
        for tag in tags:
            tag = tag.strip().lower().replace(" ", "-")
            if not tag:
                continue
            if tag not in merged:
                merged[tag] = []
            if scene_id not in merged[tag]:
                merged[tag].append(scene_id)

    return merged


# ── Tag normalization ────────────────────────────────────────────────────────


def normalize_tags(raw: str) -> list[str]:
    """Normalize a raw comma-separated tag string to canonical tags.

    Splits on commas, then for each fragment: strips surrounding
    whitespace, drops it entirely if empty after stripping, lowercases,
    and replaces internal spaces with hyphens (compound-tag convention,
    e.g. ``"Open Field"`` -> ``"open-field"``).

    This is the EXACT transformation the annotate save route did inline
    (``[t.strip().lower().replace(" ", "-") for t in raw.split(",")
    if t.strip()]``), centralized so every consumer normalizes
    identically. Behaviour is byte-identical: order preserved,
    duplicates NOT collapsed (the prior code did not dedupe), empty
    fragments dropped.

    Args:
        raw: User-entered tag string, e.g. ``"Rural,, Open Field ,"``.

    Returns:
        Normalized tag list, e.g. ``["rural", "open-field"]``.
    """
    return [t.strip().lower().replace(" ", "-") for t in raw.split(",") if t.strip()]


# ── Service-layer convenience wrappers (take a FilmContext) ──────────────────


def load_annotations(ctx: "FilmContext") -> dict:
    """Load the manual-annotations dict for ``ctx``.

    Thin pass-through to :func:`load`, keyed by the context's
    ``metadata_dir``. Returns ``{}`` when the file is absent.
    """
    return load(ctx.metadata_dir)


def save_annotations(ctx: "FilmContext", data: dict) -> Path:
    """Persist the manual-annotations dict for ``ctx`` atomically.

    Delegates to :func:`save`, which writes via a same-directory temp
    file + ``os.replace`` so a crash mid-write cannot leave a truncated
    ``manual_annotations.json``. Returns the path written.
    """
    return save(ctx.metadata_dir, data)
