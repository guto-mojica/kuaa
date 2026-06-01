"""Per-modality slate generation for the eval grading UI (E3a).

Given one parsed query from ``data/eval/m3_full_queries.yaml``, this module
calls the *real* retrieval backend for that modality and returns candidate
rows in the exact 9-key contract the ``/eval`` rows template renders (see
:func:`cinemateca.eval.seed._mock_result` — the same shape, produced live
instead of from a hand-written placeholder).

Layering: this is core (``cinemateca.*``) and MUST NOT import from ``api.*``
(enforced by import-linter); the scene_id→row join is implemented here with
``cinemateca.*`` primitives only.

E3a is hermetic and scoring-free: it produces the slate. Scoring, the CLI,
and GPU acceptance are E3b.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from cinemateca.config import Settings
from cinemateca.errors import EvalError
from cinemateca.library import Library, derive_fps, keyframe_url, load_metadata, to_smpte
from cinemateca.rhymes import find_rhymes
from cinemateca.scene_ids import scene_id_key
from cinemateca.search import Query, find

logger = logging.getLogger(__name__)

# A candidate row is the rows-template dict — exactly the 9 keys below.
CandidateRow = dict[str, Any]

_ROW_KEYS = (
    "scene_id",
    "film_slug",
    "film_title",
    "year",
    "timecode",
    "description",
    "tags",
    "score",
    "keyframe_url",
)

_VALID_TYPES = frozenset({"text", "image", "rhyme"})

# Default rhymes knobs — used when cfg.retrieval.rhymes.* is absent (e.g. a
# SimpleNamespace test cfg). Mirror config/default.yaml → retrieval.rhymes.
_DEFAULT_RHYME_DIVERSITY = 0.5
_DEFAULT_RHYME_K_CANDIDATES = 30


@dataclass(frozen=True)
class ModalQuery:
    """One parsed query from ``m3_full_queries.yaml``.

    Fields not applicable to a given ``query_type`` are ``None`` / empty:
    ``text`` is absent on rhyme queries; ``image_path`` only on image;
    ``anchor`` only on rhyme; ``w`` only on fusion. ``relevant_scene_ids``
    and ``relevance`` carry the maintainer's pre-annotation hypotheses
    (present on text queries; empty elsewhere).
    """

    id: str
    query_type: str
    text: str | None
    image_path: Path | None
    anchor: str | None
    w: float | None
    lang: str | None
    relevant_scene_ids: tuple[int, ...] = ()
    relevance: dict[str, float] = field(default_factory=dict)
    notes: str | None = None


def load_modal_queries(path: Path) -> list[ModalQuery]:
    """Load + validate ``m3_full_queries.yaml`` into a list of :class:`ModalQuery`.

    The YAML's top-level dict carries a ``queries:`` list; each entry is
    mapped to a :class:`ModalQuery` and validated per ``query_type``:

      * **text** — ``text`` present and non-empty.
      * **image** — ``image_path`` present AND the file exists on disk
        (resolved against the repo root / CWD when relative).
      * **rhyme** — ``anchor`` present AND parses as ``<slug>/<scene_id>``
        (exactly one ``/``; ``scene_id`` an int).

    Raises:
        EvalError: file missing/unreadable, malformed top-level shape,
            unknown ``query_type``, or any per-type validation failure.
    """
    import yaml  # local import — yaml is only needed for this loader

    if not path.exists():
        raise EvalError(f"eval query file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise EvalError(f"malformed eval YAML at {path}: {exc}") from exc

    if not isinstance(raw, dict) or not isinstance(raw.get("queries"), list):
        raise EvalError(f"eval YAML at {path} must be a dict with a top-level 'queries' list")

    out: list[ModalQuery] = []
    for i, entry in enumerate(raw["queries"]):
        if not isinstance(entry, dict):
            raise EvalError(f"query #{i} in {path} is not a mapping: {entry!r}")
        out.append(_parse_entry(entry, index=i, path=path))
    return out


def _parse_entry(entry: dict, *, index: int, path: Path) -> ModalQuery:
    """Map + validate one raw YAML entry into a :class:`ModalQuery`."""
    qid = str(entry.get("id") or f"query-{index}")
    qtype = entry.get("query_type")
    if qtype not in _VALID_TYPES:
        raise EvalError(
            f"query {qid!r} in {path}: unknown query_type {qtype!r} "
            f"(expected one of {sorted(_VALID_TYPES)})"
        )

    text = entry.get("text")
    text = str(text) if text is not None else None
    anchor = entry.get("anchor")
    anchor = str(anchor) if anchor is not None else None

    image_path: Path | None = None
    raw_img = entry.get("image_path")
    if raw_img is not None:
        image_path = Path(str(raw_img))

    w_raw = entry.get("w")
    w: float | None = float(w_raw) if isinstance(w_raw, (int, float)) else None

    rel_ids = tuple(int(s) for s in (entry.get("relevant_scene_ids") or []))
    rel_raw = entry.get("relevance") or {}
    relevance = {str(k): float(v) for k, v in rel_raw.items()} if isinstance(rel_raw, dict) else {}

    # ── per-type validation ──────────────────────────────────────────
    if qtype == "text":
        if not text or not text.strip():
            raise EvalError(f"query {qid!r} ({qtype}) in {path}: 'text' is required")
    elif qtype == "image":
        if image_path is None:
            raise EvalError(f"query {qid!r} (image) in {path}: 'image_path' is required")
        if not _resolve_image(image_path).exists():
            raise EvalError(
                f"query {qid!r} (image) in {path}: image_path does not exist: {image_path}"
            )
    elif qtype == "rhyme":
        if anchor is None or anchor.count("/") != 1:
            raise EvalError(
                f"query {qid!r} (rhyme) in {path}: 'anchor' must be '<slug>/<scene_id>', "
                f"got {anchor!r}"
            )
        slug, sid = anchor.split("/", 1)
        if not slug:
            raise EvalError(f"query {qid!r} (rhyme) in {path}: anchor slug is empty")
        try:
            int(sid)
        except ValueError as exc:
            raise EvalError(
                f"query {qid!r} (rhyme) in {path}: anchor scene_id is not an int: {sid!r}"
            ) from exc

    return ModalQuery(
        id=qid,
        query_type=qtype,
        text=text,
        image_path=image_path,
        anchor=anchor,
        w=w,
        lang=str(entry["lang"]) if entry.get("lang") is not None else None,
        relevant_scene_ids=rel_ids,
        relevance=relevance,
        notes=str(entry["notes"]) if entry.get("notes") is not None else None,
    )


def _resolve_image(image_path: Path) -> Path:
    """Resolve a (possibly repo-relative) image path against CWD."""
    if image_path.is_absolute():
        return image_path
    return (Path.cwd() / image_path).resolve()


# ── candidate-row builder ───────────────────────────────────────────────────


def _candidate_row(
    *,
    scene_id: int,
    film_slug: str,
    score: float,
    meta: _FilmMeta,
) -> CandidateRow:
    """Build one 9-key rows-template dict, falling back to safe defaults.

    ``meta`` carries the (optionally empty) per-film metadata lookups. When
    a scene is missing from metadata, the row still gets all 9 keys with
    safe defaults (``description=""``, ``tags=[]``, ``timecode="00:00:00"``,
    ``film_title=slug``, ``year=0``) — the row contract must always hold so
    ``/eval`` never crashes on a generated slate.
    """
    key = scene_id_key(scene_id)
    desc_entry = meta.desc_by_scene.get(key) or {}
    description = str(desc_entry.get("description", "")) if isinstance(desc_entry, dict) else ""
    kf_entry = meta.kf_by_scene.get(scene_id) or {}
    start_s = float(kf_entry.get("start_time_s") or 0.0)
    timecode = to_smpte(start_s, meta.fps) if start_s > 0 else "00:00:00"
    # Resolve the *real* served keyframe URL from the scene's stored filepath
    # (production layout: frames/scenes/keyframes_content/...), mirroring the
    # rhymes enricher. Falls back to "" when the scene has no on-disk keyframe
    # (hermetic tests / unresolvable path) — the row contract only requires a
    # string, and the template renders a placeholder for an empty src.
    keyframe_url_val = keyframe_url(str(kf_entry.get("filepath", "")), meta.data_dir) or ""
    row: CandidateRow = {
        "scene_id": int(scene_id),
        "film_slug": film_slug,
        "film_title": meta.title or film_slug,
        "year": meta.year,
        "timecode": timecode,
        "description": description,
        "tags": sorted(meta.tags_by_scene.get(key, set())),
        "score": float(score),
        "keyframe_url": keyframe_url_val,
    }
    # The 9-key contract is a self-checking invariant: every consumer
    # (the /eval rows template, E3b scoring) depends on exactly these keys.
    assert set(row) == set(_ROW_KEYS), f"candidate row key drift: {sorted(row)}"
    return row


@dataclass(frozen=True)
class _SlateFilmCtx:
    """Minimal duck-typed ``film=`` arg for :func:`cinemateca.search.find`.

    Carries exactly the attributes ``find`` (clip mode) reads — ``slug``,
    ``embeddings_dir`` (index loader) and ``metadata_dir`` (tag-filter
    path). Built from derived paths in :func:`_ctx_for`, decoupled from the
    registry-gated ``FilmContext`` so a slate works whether or not the
    slug is registered.
    """

    slug: str
    metadata_dir: Path
    embeddings_dir: Path


@dataclass(frozen=True)
class _FilmMeta:
    """Per-film metadata lookups used to fill a candidate row (cached per slug)."""

    title: str
    year: int
    fps: float
    kf_by_scene: dict[int, dict]
    desc_by_scene: dict[Any, Any]
    tags_by_scene: dict[str, set[str]]
    data_dir: Path  # /media root, for resolving a keyframe's served URL


def _empty_meta(slug: str, data_dir: Path) -> _FilmMeta:
    return _FilmMeta(
        title=slug,
        year=0,
        fps=24.0,
        kf_by_scene={},
        desc_by_scene={},
        tags_by_scene={},
        data_dir=data_dir,
    )


def _film_meta_loader(cfg: Settings, library_dir: Path):
    """Return a ``slug -> _FilmMeta`` memoised loader.

    Reads the registry (for title/year) and per-film metadata (for
    description/tags/timecode). Any failure for a slug degrades to
    :func:`_empty_meta` so the row contract still holds in hermetic tests
    that have no on-disk metadata.
    """
    library = Library(library_dir)
    # /media serves from cfg.paths.data_dir (api/server.py); keyframe filepaths
    # in metadata resolve relative to it. Fall back to library_dir.parent (the
    # data root above data/library) when the config omits an explicit data_dir.
    _paths = getattr(cfg, "paths", None)
    data_dir = Path(getattr(_paths, "data_dir", None) or library_dir.parent).resolve()
    cache: dict[str, _FilmMeta] = {}

    def _invert_tags(tag_index: dict[str, set[str]]) -> dict[str, set[str]]:
        by_scene: dict[str, set[str]] = {}
        for tag, sids in tag_index.items():
            for sid in sids:
                by_scene.setdefault(sid, set()).add(tag)
        return by_scene

    def _load(slug: str) -> _FilmMeta:
        if slug in cache:
            return cache[slug]
        # Start from the all-defaults row and override only the fields a
        # successful lookup supplies; when BOTH the registry and metadata
        # reads fail the result IS _empty_meta(slug) (keeps the docstring true).
        meta = _empty_meta(slug, data_dir)
        try:
            film = library.get_film(slug)
            year = int(film.year) if film.year is not None else 0
            meta = replace(meta, title=film.title, year=year)
        except Exception:  # noqa: BLE001 - degrade to slug/0 (incl. unregistered)
            pass
        try:
            metadata_dir = library_dir / slug / "metadata"
            kf_meta, desc_by_scene, _vis, tag_index = load_metadata(metadata_dir)
            kf_by_scene = {int(e["scene_id"]): e for e in kf_meta if "scene_id" in e}
            meta = replace(
                meta,
                fps=derive_fps(kf_meta),
                kf_by_scene=kf_by_scene,
                desc_by_scene=desc_by_scene,
                tags_by_scene=_invert_tags(tag_index),
            )
        except Exception:  # noqa: BLE001 - missing metadata is fine (hermetic)
            logger.debug("slate: no on-disk metadata for %s; using defaults", slug)
        cache[slug] = meta
        return meta

    return _load


# ── public dispatch ─────────────────────────────────────────────────────────


def generate_slate(
    *,
    query: ModalQuery,
    cfg: Settings,
    library_dir: Path,
    k: int = 9,
    film_slug: str | None = None,
) -> list[CandidateRow]:
    """Generate a candidate slate for ``query`` by calling the real backend.

    Dispatches on ``query.query_type`` to one of the ``_slate_*`` helpers,
    each of which calls the production retrieval primitive for that modality
    and maps the results into :data:`CandidateRow` dicts (descending score,
    ``k`` rows max). Films/scenes without metadata are rendered with safe
    defaults rather than raising.

    Text and image queries both route through CLIP ``find``; rhyme calls its
    dedicated primitive.

    ``film_slug`` scopes text/image search to a single film *before* the
    top-``k`` truncation. Without it the search merges all films and keeps the
    global top ``k``, so a film-scoped eval that filtered afterwards could get
    zero rows when another film dominated the global head (review #3). Ignored
    for rhyme queries, which are cross-film by definition.

    Raises:
        EvalError: unknown ``query_type`` (validation should have caught
            this at load time; re-checked defensively here).
    """
    dispatch = {
        "text": _slate_text,
        "image": _slate_image,
        "rhyme": _slate_rhyme,
    }
    helper = dispatch.get(query.query_type)
    if helper is None:
        raise EvalError(f"cannot generate slate for unknown query_type {query.query_type!r}")
    load_meta = _film_meta_loader(cfg, library_dir)
    return helper(
        query=query, cfg=cfg, library_dir=library_dir, k=k, load_meta=load_meta, film_slug=film_slug
    )


def _iter_films(library_dir: Path) -> list[str]:
    """Candidate film slugs to search, registry-first.

    Prefers slugs from the films registry (the production source of
    truth). When the registry is empty/absent — e.g. an unmigrated tree
    or a hermetic fixture — falls back to immediate subdirectories of
    ``library_dir`` (excluding the ``films.json`` sidecar), matching the
    orphan-tolerant discovery :func:`cinemateca.rhymes.find_rhymes`
    already does over ``library_dir.iterdir()``. The disk-scan fallback
    is *live* for every modality: :func:`_ctx_for` builds the ``find``
    context from derived paths (not the registry), so an unregistered
    on-disk film yields real rows rather than an empty slate.
    """
    try:
        registered = [f.slug for f in Library(library_dir).list_films()]
    except Exception:  # noqa: BLE001 - no registry yet
        registered = []
    if registered:
        return registered
    if not library_dir.exists():
        return []
    return sorted(p.name for p in library_dir.iterdir() if p.is_dir())


def _slate_find(*, q: Query, cfg, library_dir, k, load_meta, film_slug=None) -> list[CandidateRow]:
    """CLIP ``find`` over ``q`` per film, merged by descending score.

    Shared by the text and image dispatch paths — both call ``find`` in
    CLIP mode and differ only in the :class:`Query` object built.

    ``film_slug`` restricts the search to that single film, so the top-``k``
    truncation happens within the scoped film rather than across the whole
    library (review #3).
    """
    rows: list[CandidateRow] = []
    slugs = [film_slug] if film_slug else _iter_films(library_dir)
    for slug in slugs:
        ctx = _ctx_for(library_dir, slug)
        if ctx is None:
            continue
        result = find(q, film=ctx, mode="clip", top_k=k, cfg=cfg)
        meta = load_meta(slug)
        for hit in result.hits:
            # Only hit.scene_id + hit.score are load-bearing here. Description,
            # tags, year and timecode are re-read from per-film metadata in
            # _candidate_row because Hit carries none of those (Hit.description
            # exists but tags/year/timecode do not — see search.types.Hit), so
            # we go to metadata for all four to keep the 9-key row consistent.
            rows.append(
                _candidate_row(scene_id=hit.scene_id, film_slug=slug, score=hit.score, meta=meta)
            )
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:k]


def _slate_text(*, query, cfg, library_dir, k, load_meta, film_slug=None) -> list[CandidateRow]:
    """Text query → CLIP ``find(Query.of_text(...))`` per film, merged."""
    q = Query.of_text(query.text or "")
    return _slate_find(
        q=q, cfg=cfg, library_dir=library_dir, k=k, load_meta=load_meta, film_slug=film_slug
    )


def _slate_image(*, query, cfg, library_dir, k, load_meta, film_slug=None) -> list[CandidateRow]:
    """Image query → CLIP-only ``find(Query.image(...))`` per film, merged."""
    assert query.image_path is not None  # validated at load time
    q = Query.image(_resolve_image(query.image_path))
    return _slate_find(
        q=q, cfg=cfg, library_dir=library_dir, k=k, load_meta=load_meta, film_slug=film_slug
    )


def _slate_rhyme(*, query, cfg, library_dir, k, load_meta, film_slug=None) -> list[CandidateRow]:
    """Rhyme query → cross-film ``find_rhymes`` from the parsed anchor.

    ``film_slug`` is accepted for a uniform dispatch signature but ignored —
    rhymes are cross-film by definition (``cross_film_only=True``).
    """
    assert query.anchor is not None  # validated at load time
    slug, sid_s = query.anchor.split("/", 1)
    anchor_scene_id = int(sid_s)
    rhymes_cfg = getattr(getattr(cfg, "retrieval", None), "rhymes", None)
    lambda_div = getattr(rhymes_cfg, "diversity", _DEFAULT_RHYME_DIVERSITY)
    k_candidates = getattr(rhymes_cfg, "k_candidates", _DEFAULT_RHYME_K_CANDIDATES)
    rhymes = find_rhymes(
        library_dir,
        slug,
        anchor_scene_id,
        top_n=k,
        cross_film_only=True,
        lambda_diversity=float(lambda_div),
        k_candidates=int(k_candidates),
    )
    rows: list[CandidateRow] = []
    for r in rhymes:
        meta = load_meta(r.film_slug)
        rows.append(
            _candidate_row(
                scene_id=int(r.scene_id), film_slug=r.film_slug, score=float(r.score), meta=meta
            )
        )
    return rows[:k]


def _ctx_for(library_dir: Path, slug: str) -> _SlateFilmCtx | None:
    """Build a per-film context for CLIP ``find`` from derived paths.

    ``find`` (clip mode) duck-types its ``film=`` arg on ``.slug`` /
    ``.embeddings_dir`` (the index loader) plus ``.metadata_dir`` (the
    tag-filter path) — see :func:`cinemateca.search.find`. We construct
    those paths directly from ``library_dir/<slug>/...`` rather than going
    through the registry-gated ``FilmContext.from_paths`` /
    ``Library.context``, so this works for an on-disk-but-*unregistered*
    film just as well as a registered one (the disk-scan fallback in
    :func:`_iter_films` would otherwise yield slugs that produced zero
    rows). Title/year for the row still come from the registry when present
    (via :func:`_film_meta_loader`) and degrade to ``slug`` / ``0`` when not.

    A traversal slug (``slug != Path(slug).name``) degrades to ``None`` →
    film skipped (mirrors the guard in ``FilmContext.from_paths``).
    """
    if not slug or slug != Path(slug).name:
        return None
    film_dir = library_dir / slug
    return _SlateFilmCtx(
        slug=slug,
        metadata_dir=film_dir / "metadata",
        embeddings_dir=film_dir / "embeddings",
    )


__all__ = [
    "CandidateRow",
    "ModalQuery",
    "generate_slate",
    "load_modal_queries",
]
