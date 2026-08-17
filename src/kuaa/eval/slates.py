"""Per-modality slate generation for the eval grading UI (E3a).

Given one parsed query from a query set (``data/eval/corpus01_queries.yaml``
is the current one), this module
calls the *real* retrieval backend for that modality and returns candidate
rows in the exact 9-key contract the ``/eval`` rows template renders (see
:func:`kuaa.eval.seed._mock_result` — the same shape, produced live
instead of from a hand-written placeholder).

Layering: this is core (``kuaa.*``) and MUST NOT import from ``api.*``
(enforced by import-linter); the scene_id→row join is implemented here with
``kuaa.*`` primitives only.

E3a is hermetic and scoring-free: it produces the slate. Scoring, the CLI,
and GPU acceptance are E3b.
"""

from __future__ import annotations

import copy
import hashlib
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from kuaa.config import Settings
from kuaa.errors import EvalError
from kuaa.eval.registry import CLIP_ONLY, POOL_VARIANTS, RetrieverVariant
from kuaa.library import Library, derive_fps, keyframe_url, load_metadata, to_smpte
from kuaa.retrieval.hybrid import DEFAULT_RRF_K
from kuaa.rhymes import find_rhymes
from kuaa.scene_ids import scene_id_key
from kuaa.search import Query, find
from kuaa.search._aggregate.fusion import fuse_global_rrf

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


def _known_search_keys(search: Any) -> set[str] | None:
    """Field names ``search`` accepts, or ``None`` when they can't be known.

    Pydantic models declare ``model_fields``; the duck-typed namespaces the
    test suite passes expose ``__dict__``. Anything else (a mock, a proxy)
    returns ``None``, which the caller reads as "cannot validate here".
    """
    fields = getattr(type(search), "model_fields", None)
    if isinstance(fields, dict):
        return set(fields)
    attrs = getattr(search, "__dict__", None)
    if isinstance(attrs, dict):
        return set(attrs)
    return None


def _cfg_with(cfg: Any, **search_overrides: Any) -> Any:
    """Read-through copy of ``cfg`` with ``cfg.search`` fields replaced.

    Returns ``cfg`` untouched when there is nothing to override, so the common
    path allocates nothing. Handles both the pydantic ``Settings`` and the
    duck-typed ``SimpleNamespace`` configs the test suite passes.

    Every override key is checked against the settings model first.
    ``model_copy(update=...)`` does **not** validate, so a typo'd key used to be
    set silently and the variant quietly became a duplicate of the one it was
    meant to differ from — the exact failure this function exists to prevent,
    reintroduced by the mechanism chosen to implement it.

    Raises:
        EvalError: an override names a field ``cfg.search`` does not have.
    """
    if not search_overrides or cfg is None:
        return cfg
    search = getattr(cfg, "search", None)
    if search is None:
        # No section to copy — synthesise one. Returning cfg untouched here
        # would make the override a silent no-op, so ``hybrid_no_metadata``
        # would quietly be a second plain ``hybrid`` and the pool would look
        # like it spanned five retrievers while spanning four.
        new_cfg = copy.copy(cfg)
        new_cfg.search = SimpleNamespace(**search_overrides)
        return new_cfg
    known = _known_search_keys(search)
    if known is not None:
        unknown = sorted(set(search_overrides) - known)
        if unknown:
            raise EvalError(
                f"unknown cfg.search override(s) {unknown} on {type(search).__name__} — "
                f"a variant that overrides a field the config does not have is "
                f"indistinguishable from the variant it was meant to differ from"
            )
    if hasattr(search, "model_copy") and hasattr(cfg, "model_copy"):
        return cfg.model_copy(update={"search": search.model_copy(update=dict(search_overrides))})
    new_search = copy.copy(search)
    for key, value in search_overrides.items():
        setattr(new_search, key, value)
    new_cfg = copy.copy(cfg)
    new_cfg.search = new_search
    return new_cfg


# Default rhymes knobs — used when cfg.retrieval.rhymes.* is absent (e.g. a
# SimpleNamespace test cfg). Mirror config/default.yaml → retrieval.rhymes.
_DEFAULT_RHYME_DIVERSITY = 0.5
_DEFAULT_RHYME_K_CANDIDATES = 30


@dataclass(frozen=True)
class ModalQuery:
    """One parsed query from a query-set YAML.

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


def load_modal_queries(path: Path, *, only_types: set[str] | None = None) -> list[ModalQuery]:
    """Load + validate a query-set YAML into a list of :class:`ModalQuery`.

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
        if only_types is not None and entry.get("query_type") not in only_types:
            continue
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


def _scene_timecode(start_s: float, end_s: float, fps: float) -> str:
    """``start – end (duration)`` for one scene; the start alone when unknown.

    The row shows the scene's **extent**, not just where it opens, because a
    grader comparing a description against a thumbnail needs to know how much
    scene sits between them. The thumbnail is the middle keyframe and the
    stamp is the start, so on a long scene they are legitimately seconds
    apart — without the extent, that reads as the row being wrong and sends
    the grader to the video file to check. Half this library's scenes run over
    5s and the 90th percentile is 31s, so it is not a rare case.

    Falls back to the bare start when ``end_s`` is missing or not after it:
    an unknown extent must not be rendered as a zero-length one.
    """
    start = to_smpte(start_s, fps)
    if end_s <= start_s:
        return start
    duration = end_s - start_s
    # One decimal under 10s — a "0s" scene reads as broken metadata, and at
    # this end of the range the fraction is the difference between a cut and
    # a held shot.
    length = f"{duration:.1f}s" if duration < 10 else f"{round(duration)}s"
    return f"{start} – {to_smpte(end_s, fps)} ({length})"


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
    timecode = (
        _scene_timecode(
            float(kf_entry.get("start_time_s") or 0.0),
            float(kf_entry.get("end_time_s") or 0.0),
            meta.fps,
        )
        if kf_entry
        else "00:00:00"
    )
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
    # A ``raise``, not an ``assert``: this is the sole enforcement of the
    # contract, and ``python -O`` strips asserts.
    if set(row) != set(_ROW_KEYS):
        raise ValueError(f"candidate row key drift: {sorted(row)}")
    return row


@dataclass(frozen=True)
class _SlateFilmCtx:
    """Minimal duck-typed ``film=`` arg for :func:`kuaa.search.find`.

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


def _representative_keyframes(kf_meta: list) -> dict[int, dict]:
    """``scene_id -> the scene's MIDDLE keyframe row``.

    Scene detection writes ``keyframes_per_scene`` rows per scene (3 by
    default), and the middle one is this project's representative frame
    everywhere it is chosen deliberately — ``annotations.scenes.
    build_scene_list``, ``preprocess.service._grouped_scenes``, and the
    describer itself (``llm.keyframes: middle``, so ``scene_descriptions``
    describes ``scene_NNNN_kf_02``).

    This map used to be a dict comprehension over every row, which is
    last-write-wins: it selected ``kf_03``, near the END of the scene, while
    the description beside it in the same row described ``kf_02``. Nothing
    was inconsistent about the data — the row was assembling two different
    moments and presenting them as one. A grader checking the thumbnail
    against the description, or against the video at the row's timecode,
    finds them a few seconds apart, and the gap scales with scene duration
    (library median 4.7s, p90 31.5s), which is why it reads as intermittent.

    Sorted by ``keyframe_id`` rather than trusting file order, so the
    "middle" is the middle of the scene and not of however the rows happened
    to be written.
    """
    groups: dict[int, list[dict]] = {}
    for entry in kf_meta:
        if not isinstance(entry, dict) or entry.get("scene_id") is None:
            continue
        try:
            sid = int(entry["scene_id"])
        except (TypeError, ValueError):
            continue
        groups.setdefault(sid, []).append(entry)
    return {
        sid: sorted(group, key=lambda e: str(e.get("keyframe_id", "")))[len(group) // 2]
        for sid, group in groups.items()
    }


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
            kf_by_scene = _representative_keyframes(kf_meta)
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


def rank_candidates(
    *,
    query: ModalQuery,
    cfg: Settings,
    library_dir: Path,
    k: int = 9,
    film_slug: str | None = None,
) -> list[CandidateRow]:
    """Candidates for ``query`` in the retrieval system's own order, scores intact.

    This is the **measurement** surface. Dispatches on ``query.query_type`` to
    one of the ``_slate_*`` helpers, each of which calls the production
    retrieval primitive for that modality and maps the results into
    :data:`CandidateRow` dicts (``k`` rows max). Films/scenes without metadata
    are rendered with safe defaults rather than raising.

    Text queries are **pooled** across every retriever variant (see
    :func:`pool_candidates`); image queries are CLIP-only and rhyme calls its
    dedicated primitive, neither having a competing retriever to union with.

    Never hand the result to a grader — position is the system's opinion and
    ``score`` is its confidence. :func:`generate_slate` is the grader-facing
    surface, and it is blind by construction rather than by argument: a
    boolean whose wrong value silently produces wrong numbers is not a safe
    thing to own. (It did: every image and rhyme metric this project has
    published was computed over a seeded shuffle.)

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


def generate_slate(
    *,
    query: ModalQuery,
    cfg: Settings,
    library_dir: Path,
    k: int = 9,
    film_slug: str | None = None,
    blind_seed: str = "",
) -> list[CandidateRow]:
    """Generate a grader-facing candidate slate — always blinded.

    :func:`rank_candidates` followed by :func:`blind`. There is no argument
    that turns the blinding off; measurement tooling calls
    :func:`rank_candidates` directly and says so at its call site.
    """
    rows = rank_candidates(query=query, cfg=cfg, library_dir=library_dir, k=k, film_slug=film_slug)
    return blind(rows, seed=f"{blind_seed}:{query.id}")


def _blind_order_key(row: CandidateRow, *, seed: str) -> bytes:
    """Stable per-candidate sort key: hash of ``(seed, film_slug, scene_id)``."""
    material = f"{seed}\x00{row.get('film_slug', '')}\x00{scene_id_key(row.get('scene_id', ''))}"
    return hashlib.blake2b(material.encode("utf-8"), digest_size=16).digest()


def blind(rows: list[CandidateRow], *, seed: str) -> list[CandidateRow]:
    """Strip retrieval provenance from the *presentation* of ``rows``.

    Two leaks, both in the data rather than the template:

    * **Order.** Rows arrive in the proposing retriever's rank order, so
      position *is* the system's opinion. A grader who reads position as a
      hint grades the system instead of the scene.
    * **Score.** ``rows.html`` renders ``score`` as a number and a progress
      bar, which broadcasts the retriever's confidence.

    Order is a sort on a per-candidate hash of ``(seed, film_slug, scene_id)``,
    not a shuffle of the list. Both give the same guarantee for an unchanged
    pool — a grader resuming a run meets the same queue — but a list shuffle
    repermutes *everything* when one candidate is added or removed, which
    breaks that promise on exactly the edit that motivates it. Keying on the
    candidate instead means a pool edit moves only the candidates it touched,
    so a variant change can be re-graded as a diff rather than a full pass.

    ``pool`` survives untouched: the template ignores unknown keys, and it is
    what makes per-leg scoring possible after grading.
    """
    ordered = sorted(rows, key=lambda row: _blind_order_key(row, seed=seed))
    return [{**row, "score": None} for row in ordered]


def _iter_films(library_dir: Path) -> list[str]:
    """Candidate film slugs to search, registry-first.

    Prefers slugs from the films registry (the production source of
    truth). When the registry is empty/absent — e.g. an unmigrated tree
    or a hermetic fixture — falls back to immediate subdirectories of
    ``library_dir`` (excluding the ``films.json`` sidecar), matching the
    orphan-tolerant discovery :func:`kuaa.rhymes.find_rhymes`
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


def _merge_across_films(
    per_film: dict[str, list[CandidateRow]], *, k: int, rrf_k: int = DEFAULT_RRF_K
) -> list[CandidateRow]:
    """Interleave per-film ranked lists into one cross-film ranking.

    Rank-based, via :func:`fuse_global_rrf` on ``(slug, scene_id)`` keys — the
    same primitive and the same key the cross-film ``aggregate`` path uses.
    Sorting a flat list by raw ``score`` instead is only defensible for CLIP:
    BM25 scores carry per-film IDF and RRF fused scores are per-film rank
    transforms, so neither is comparable between films. The previous sort made
    the pool's cross-film membership a function of which film happened to
    produce larger numbers.

    Each film's list gets equal weight, which makes the result **round-robin
    by sorted slug**: every film's rank-1 candidate scores ``1/(rrf_k+1)``, so
    all rank-1s precede all rank-2s, and within a rank the stable sort
    preserves the sorted-slug insertion order. That interleave is part of the
    pool's semantics — it decides which candidates survive the ``k`` cut — so
    it is specified here and pinned by test rather than left to emerge.

    ``score`` on the returned rows stays the retriever's own value: it is
    meaningful *within* a film and is what the measurement path reports.
    Position, not score, is the cross-film ranking.

    **``k`` must be at least the number of films.** Below that the cut lands
    mid-rotation and the last-sorting slugs are dropped from every call — by
    name, not by relevance. It is a property of ``k`` and the corpus rather
    than of any retriever, so no per-variant check can see it;
    :func:`kuaa.eval.composition.analyse_pool` checks it against the searched
    film list and fails the run.

    The interleave is deliberately blind to how strongly a film matched: a
    query answered entirely by one film still gets one candidate per film at
    each rank. That is the price of not comparing scores across films, and it
    is why the ``pool`` rank values encode slug position rather than
    confidence — they identify *which* variants proposed a scene, which is
    what the composition report and per-leg recall need, and they are not a
    per-variant ranking to score.
    """
    by_key: dict[str, CandidateRow] = {}
    weighted: list[tuple[list[tuple[str, float]], float]] = []
    for slug in sorted(per_film):
        ranked: list[tuple[str, float]] = []
        for row in per_film[slug]:
            key = f"{slug}/{scene_id_key(row['scene_id'])}"
            by_key[key] = row
            ranked.append((key, float(row["score"])))
        weighted.append((ranked, 1.0))
    fused = fuse_global_rrf(weighted, k_rrf=rrf_k)
    return [by_key[key] for key, _score in fused[:k]]


def _slate_find(
    *, q: Query, cfg, library_dir, k, load_meta, film_slug=None, variant: RetrieverVariant | None
) -> list[CandidateRow]:
    """One retriever variant's ``find`` over ``q`` per film, interleaved by rank.

    ``variant`` selects the retriever; ``None`` means CLIP with stock config,
    which is what the image path wants (``find`` forces CLIP for image queries
    regardless of ``mode``).

    ``film_slug`` restricts the search to that single film, so the top-``k``
    truncation happens within the scoped film rather than across the whole
    library.

    Per-film order is ``find``'s own — which, for the rerank variant, is the
    cross-encoder's order. The previous re-sort by ``hit.score`` discarded it
    (``rerank`` writes ``Hit.rerank_score`` and leaves ``Hit.score`` alone),
    so the reranked variant paid for a cross-encoder pass and then handed back
    the ranking it started from.
    """
    variant = variant or CLIP_ONLY
    call_cfg = _cfg_with(cfg, **variant.cfg_overrides)
    per_film: dict[str, list[CandidateRow]] = {}
    slugs = [film_slug] if film_slug else _iter_films(library_dir)
    for slug in slugs:
        ctx = _ctx_for(library_dir, slug)
        if ctx is None:
            continue
        result = find(
            q,
            film=ctx,
            mode=variant.mode,
            top_k=k,
            rerank=variant.rerank,
            cfg=call_cfg,
        )
        meta = load_meta(slug)
        # Only hit.scene_id + hit.score are load-bearing here. Description,
        # tags, year and timecode are re-read from per-film metadata in
        # _candidate_row because Hit carries none of those (Hit.description
        # exists but tags/year/timecode do not — see search.types.Hit), so
        # we go to metadata for all four to keep the 9-key row consistent.
        per_film[slug] = [
            _candidate_row(scene_id=hit.scene_id, film_slug=slug, score=hit.score, meta=meta)
            for hit in result.hits
        ]
    return _merge_across_films(per_film, k=k, rrf_k=_rrf_k_from(cfg))


def _rrf_k_from(cfg: Any) -> int:
    """``cfg.search.bm25.rrf_k`` when present, else the shipped default."""
    bm25_cfg = getattr(getattr(cfg, "search", None), "bm25", None)
    try:
        return int(getattr(bm25_cfg, "rrf_k", DEFAULT_RRF_K))
    except (TypeError, ValueError):
        return DEFAULT_RRF_K


def pool_candidates(
    *, q: Query, cfg, library_dir, k, load_meta, film_slug=None, variants=POOL_VARIANTS
) -> list[CandidateRow]:
    """Union every variant's top-``k``, keyed by (film, scene), ranks recorded.

    Relevance judgments are only valid for the system that produced the pool
    they were drawn from. Grading a CLIP-only slate yields labels that are
    biased against BM25, hybrid and the reranker — exactly the comparisons the
    eval exists to make — so the pool spans every retriever under comparison
    and the judgments outlive any one of them.

    Each row carries ``pool``: ``{variant_name: rank_it_proposed_at}``. That is
    what makes per-leg scoring possible after grading; without it a graded pool
    can only be scored as a whole.

    Output order is dict insertion order: every variant's candidates in the
    order :func:`_merge_across_films` produced them, variants in registry
    order, each candidate held at the position of the *first* variant to
    propose it. That is load-bearing — it is the order :func:`blind` is handed
    and the order a truncated pool would keep — so it is pinned by test.
    """
    pooled: dict[str, CandidateRow] = {}
    for variant in variants:
        rows = _slate_find(
            q=q,
            cfg=cfg,
            library_dir=library_dir,
            k=k,
            load_meta=load_meta,
            film_slug=film_slug,
            variant=variant,
        )
        for rank, row in enumerate(rows, start=1):
            key = f"{row['film_slug']}/{scene_id_key(row['scene_id'])}"
            seen = pooled.get(key)
            if seen is None:
                row["pool"] = {variant.name: rank}
                pooled[key] = row
            else:
                seen["pool"][variant.name] = rank
    return list(pooled.values())


def _slate_text(*, query, cfg, library_dir, k, load_meta, film_slug=None) -> list[CandidateRow]:
    """Text query → the union of every retriever variant's top-``k``."""
    q = Query.of_text(query.text or "")
    return pool_candidates(
        q=q, cfg=cfg, library_dir=library_dir, k=k, load_meta=load_meta, film_slug=film_slug
    )


def _slate_image(*, query, cfg, library_dir, k, load_meta, film_slug=None) -> list[CandidateRow]:
    """Image query → CLIP-only ``find(Query.image(...))`` per film, merged.

    Not pooled: ``find`` forces CLIP for image queries whatever ``mode`` says,
    so there is no competing retriever to union with.
    """
    assert query.image_path is not None  # validated at load time
    q = Query.image(_resolve_image(query.image_path))
    return _slate_find(
        q=q,
        cfg=cfg,
        library_dir=library_dir,
        k=k,
        load_meta=load_meta,
        film_slug=film_slug,
        variant=CLIP_ONLY,
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


# ── thin persistence ────────────────────────────────────────────────────────
#
# What a graded pool must pin is WHICH scenes were put in front of the grader
# and which retriever proposed each — ``(film_slug, scene_id, pool)``. The
# other six keys are presentation: they are re-read from per-film metadata by
# the same builder that produced them, so persisting them costs an order of
# magnitude in file size and pins a caption that was never the unit of
# judgment. The grader judges the scene; the description is context that may
# legitimately be regenerated under a stable pool.

_THIN_KEYS = ("scene_id", "film_slug", "pool")


def thin_rows(rows: list[CandidateRow]) -> list[dict]:
    """Reduce candidate rows to the provenance the grades depend on.

    Order is preserved because it is load-bearing: rows reach disk already
    shuffled by :func:`blind`, and a grader resuming a run must meet the same
    queue. ``score`` is dropped rather than carried as ``None`` — a blinded
    row has none, and an unblinded one must not reach a grader.
    """
    out: list[dict] = []
    for row in rows:
        thin = {k: row[k] for k in _THIN_KEYS if k in row}
        out.append(thin)
    return out


def hydrate_rows(rows: list[dict], *, cfg: Settings, library_dir: Path) -> list[CandidateRow]:
    """Rebuild full rows-template rows from thin ones, preserving order.

    A row that already carries the full key set passes through with its
    ``score`` blanked, so fat slates written before thinning — and the mock
    rows from ``kuaa.eval.seed`` — keep rendering without a migration.

    ``score`` is ``None`` on **every** returned row, whichever path it took: a
    persisted slate is a graded pool's candidate list, and a score here is the
    retriever's opinion that :func:`blind` exists to withhold. The fat-row
    passthrough used to return ``score`` intact while this docstring promised
    otherwise — the one re-entry point where a stored score could reach the
    page.
    """
    load_meta = _film_meta_loader(cfg, library_dir)
    out: list[CandidateRow] = []
    for row in rows:
        if set(row) >= set(_ROW_KEYS):
            out.append(cast(CandidateRow, {**row, "score": None}))
            continue
        slug = str(row.get("film_slug", ""))
        try:
            scene_id = int(row.get("scene_id", 0))
        except (TypeError, ValueError):
            continue
        full = _candidate_row(scene_id=scene_id, film_slug=slug, score=0.0, meta=load_meta(slug))
        full["score"] = None
        if "pool" in row:
            full["pool"] = row["pool"]
        out.append(full)
    return out


def _ctx_for(library_dir: Path, slug: str) -> _SlateFilmCtx | None:
    """Build a per-film context for CLIP ``find`` from derived paths.

    ``find`` (clip mode) duck-types its ``film=`` arg on ``.slug`` /
    ``.embeddings_dir`` (the index loader) plus ``.metadata_dir`` (the
    tag-filter path) — see :func:`kuaa.search.find`. We construct
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
    "CLIP_ONLY",
    "POOL_VARIANTS",
    "CandidateRow",
    "ModalQuery",
    "RetrieverVariant",
    "generate_slate",
    "hydrate_rows",
    "load_modal_queries",
    "pool_candidates",
    "rank_candidates",
    "thin_rows",
]
