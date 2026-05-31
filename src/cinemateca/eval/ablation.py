"""Proxy-first ablation table — the WS-4 portfolio centrepiece (E2b).

This module produces a retriever-variant ablation table that is **publishable
with zero human grades**. Every row is scored on a *common query set* with the
*same* proxy labels (apples-to-apples), the proxy signal is named in a ``Proxy``
column, and any backend that is not wired renders a literal ``pending (...)``
cell — never a fabricated or zero number.

Design (spec §6 E2)
-------------------
The common query set is the **15 text queries** from
``data/eval/m3_full_queries.yaml``; every one carries the maintainer's
pre-curator hypothesis (``relevant_scene_ids`` / ``relevance``), so
:func:`cinemateca.eval.proxy.proxy_labels` returns ``"HY"`` for all of them and
the whole table is **one honesty tier** — no tautological pseudo-relevance, no
structurally-zero rhyme row blended into the average.

Rows:

================  ==========================================================  =====
row               how                                                         proxy
================  ==========================================================  =====
``CLIP``          :func:`run_retrieval_eval` (SigLIP2 default index)          HY
``BM25``          :func:`run_retrieval_eval` ``retriever="bm25"``             HY
``hybrid``        :func:`run_retrieval_eval` ``retriever="hybrid"``           HY
``hybrid+rerank`` production ``find(mode="hybrid", rerank=...)`` ± C5         HY
``fusion``        :func:`cinemateca.search.fusion.search_fusion` per query    HY
``multilingual``  C8: OpenCLIP index vs the SigLIP2 ``CLIP`` row              HY
================  ==========================================================  =====

The reranker only scores **text** queries (it reads ``query.text``), which is
the common set, so the rerank delta is well-defined. The rerank row uses the
production :func:`cinemateca.search.find` for *both* its hybrid base
(``rerank=False``) and its reranked variant (``rerank=True``); the plain
``hybrid`` row above is the harness's own ``run_retrieval_eval`` path. They are
two different hybrid implementations, so the table footnote states the rerank
delta is measured on ``find``'s hybrid, not on the harness hybrid row.

The ``multilingual`` row is the C8 backend comparison: it re-runs the CLIP path
against the on-disk OpenCLIP index (``keyframe_embeddings.clip_openclip.npy`` +
``index_mapping.clip_openclip.json``, both present per film) with the
``clip_openclip`` text encoder, so the delta to the SigLIP2 ``CLIP`` row shows
the multilingual upgrade's effect on the (PT/EN) query mix. Both indexes exist
on disk, so the switch is a config-copy mutation — no fragile plumbing.

Layering: this is core (``cinemateca.*``); it MUST NOT import ``api.*``
(import-linter). The reranker (:func:`cinemateca.search.rerank.rerank` via
``find``) and :func:`cinemateca.search.fusion.search_fusion` are all
``cinemateca``-side.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cinemateca.config import Settings
from cinemateca.errors import EvalError
from cinemateca.eval.datasets import EvaluationDataset, QueryCase
from cinemateca.eval.metrics import evaluate_query, summarize_results
from cinemateca.eval.proxy import proxy_labels
from cinemateca.eval.retrieval import RetrievalRun, run_retrieval_eval
from cinemateca.eval.slates import ModalQuery, _normalise_clip_mapping, _NullEncoder
from cinemateca.scene_ids import scene_id_key

logger = logging.getLogger(__name__)

# Metric keys rendered as table columns, in display order. Maps the
# ``summarize_results`` keys to their published column headers.
_METRIC_COLUMNS: tuple[tuple[str, str], ...] = (
    ("recall_at_5", "Recall@5"),
    ("recall_at_10", "Recall@10"),
    ("mrr", "MRR"),
    ("ndcg_at_10", "nDCG@10"),
)

# On-disk OpenCLIP (C8 baseline) index filenames — present per film alongside
# the SigLIP2 default. Mirrors the ``.clip_openclip`` suffix the M3 SigLIP2
# rollout left for rollback.
_OPENCLIP_EMB_FILENAME = "keyframe_embeddings.clip_openclip.npy"
_OPENCLIP_MAP_FILENAME = "index_mapping.clip_openclip.json"

# Default fusion visual↔audio weight for the fusion row (matches the
# ``search_fusion`` default and the M3 UI midpoint).
_FUSION_WEIGHT = 0.5


@dataclass(frozen=True)
class AblationRowConfig:
    """One row of the ablation table.

    Attributes:
        name: published row label (e.g. ``"hybrid+rerank"``).
        retriever: which retriever mechanism to run —
            ``"clip" | "bm25" | "hybrid" | "fusion" | "multilingual"``.
        proxy: the proxy signal used for the row's labels — ``"KI" | "PR" |
            "HY"`` (the whole launch table is ``"HY"``; the field exists so a
            future mixed table can segregate tiers).
        rerank: when ``True`` the row applies the C5 cross-encoder on top of a
            ``find``-based hybrid base (only meaningful with
            ``retriever == "hybrid"``).
        pending_reason: when set, the row is rendered ``pending (<reason>)`` and
            :func:`run_ablation` does not attempt to compute it. Used for a row
            whose backend is not wired (e.g. ``"C5"`` / ``"C8"``).
    """

    name: str
    retriever: str
    proxy: str = "HY"
    rerank: bool = False
    pending_reason: str | None = None


@dataclass
class AblationTable:
    """Rendered ablation result — rows paired with their metric dicts.

    ``rows`` is a list of ``(AblationRowConfig, metrics | None)``: a ``None``
    metrics value marks a ``pending`` row (its cells render ``pending
    (<reason>)``). ``corpus`` and ``common_query_set`` populate the methodology
    banner. ``validated_label`` overrides the proxy-methodology banner when human
    grades were used (set by the caller when ``graded_labels`` is provided to
    :func:`run_ablation`).
    """

    rows: list[tuple[AblationRowConfig, dict[str, float | int] | None]] = field(
        default_factory=list
    )
    corpus: str = ""
    common_query_set: str = ""
    # Optional per-row footnotes keyed by row name (e.g. the rerank-base note).
    footnotes: dict[str, str] = field(default_factory=dict)
    # When set, the banner flips from proxy wording to this human-validated label.
    validated_label: str | None = None

    def _banner(self) -> list[str]:
        """The methodology banner: KI/PR/HY definitions + corpus + caveat."""
        if self.validated_label:
            return [
                f"**Human-validated methodology.** {self.validated_label} "
                "Every row below is scored on a common query set with the **same** "
                "human grades, so the comparison is apples-to-apples. Proxy signals:",
            ]
        return [
            "**Proxy methodology.** These are **proxy metrics**, not human-graded "
            "ground truth — they upgrade to human-validated numbers when curator "
            "grades land (WS-4 E5). Every row below is scored on a common query "
            "set with the **same** proxy labels, so the comparison is "
            "apples-to-apples. Proxy signals:",
            "",
            "- **HY (Hypothesis)** — the maintainer's pre-curator "
            "`relevant_scene_ids` / `relevance` from the query file. Best-guess "
            "relevant scenes recorded before any grading session.",
            "- **KI (Known-Item)** — the single anchor scene a query came from "
            "(image keyframe / rhyme anchor). Not used in this table.",
            "- **PR (Pseudo-Relevance)** — a reference retriever's top-1 treated "
            "as relevant (relative agreement). Not used in this table.",
            "",
            f"**Corpus.** {self.corpus or 'demo library'}.",
            f"**Common query set.** {self.common_query_set or 'text queries'} — "
            "all labelled **HY**.",
            "",
        ]

    def to_markdown(self) -> str:
        """Render a rows × {Recall@5, Recall@10, MRR, nDCG@10} pipe table.

        A ``Proxy`` column records each row's signal; a pending row renders
        ``pending (<reason>)`` across its four metric cells (never a number).
        The methodology banner precedes the table; per-row footnotes (if any)
        follow it.
        """
        headers = ["Retriever", "Proxy", *[label for _key, label in _METRIC_COLUMNS]]
        sep = ["---", "---", *["---:" for _ in _METRIC_COLUMNS]]
        lines = self._banner()
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(sep) + " |")

        for row_cfg, metrics in self.rows:
            cells = [row_cfg.name, row_cfg.proxy]
            if metrics is None:
                reason = row_cfg.pending_reason or "not wired"
                pending = f"pending ({reason})"
                cells.extend([pending for _ in _METRIC_COLUMNS])
            else:
                for key, _label in _METRIC_COLUMNS:
                    value = metrics.get(key)
                    cells.append(f"{float(value):.3f}" if value is not None else "pending (n/a)")
            lines.append("| " + " | ".join(cells) + " |")

        if self.footnotes:
            lines.append("")
            for name, note in self.footnotes.items():
                lines.append(f"> **{name}.** {note}")

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Default row configs.
# ─────────────────────────────────────────────────────────────────────────────

# Full set (rerank row REAL — uses production find ± C5).
DEFAULT_ABLATION_CONFIGS: tuple[AblationRowConfig, ...] = (
    AblationRowConfig(name="CLIP", retriever="clip", proxy="HY"),
    AblationRowConfig(name="BM25", retriever="bm25", proxy="HY"),
    AblationRowConfig(name="hybrid", retriever="hybrid", proxy="HY"),
    AblationRowConfig(name="hybrid+rerank", retriever="hybrid", proxy="HY", rerank=True),
    AblationRowConfig(name="fusion", retriever="fusion", proxy="HY"),
    AblationRowConfig(name="multilingual", retriever="multilingual", proxy="HY"),
)

# No-rerank variant — the rerank row is pending (C5) so the table is produced
# without paying the cross-encoder cost (and the committed --no-rerank doc is
# honest about which rows are real).
DEFAULT_ABLATION_CONFIGS_NO_RERANK: tuple[AblationRowConfig, ...] = (
    AblationRowConfig(name="CLIP", retriever="clip", proxy="HY"),
    AblationRowConfig(name="BM25", retriever="bm25", proxy="HY"),
    AblationRowConfig(name="hybrid", retriever="hybrid", proxy="HY"),
    AblationRowConfig(
        name="hybrid+rerank", retriever="hybrid", proxy="HY", rerank=True, pending_reason="C5"
    ),
    AblationRowConfig(name="fusion", retriever="fusion", proxy="HY"),
    AblationRowConfig(name="multilingual", retriever="multilingual", proxy="HY"),
)

# Footnotes attached to the rendered table when the matching row is present.
_ROW_FOOTNOTES: dict[str, str] = {
    "hybrid+rerank": (
        "Rerank delta is measured on the production `find(mode=\"hybrid\")` base "
        "(± the C5 bge-reranker-v2-m3 cross-encoder), which is a different hybrid "
        "implementation from the harness `hybrid` row above — compare the rerank "
        "row to the `find` hybrid base it sits on, not to the harness `hybrid` row."
    ),
    "multilingual": (
        "C8 baseline: the CLIP path re-run against the on-disk OpenCLIP index "
        "(`keyframe_embeddings.clip_openclip.npy`) with the `clip_openclip` text "
        "encoder. The delta to the SigLIP2 `CLIP` row is the multilingual "
        "upgrade's effect on the PT/EN query mix."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Common query set → HY-labelled text dataset.
# ─────────────────────────────────────────────────────────────────────────────


def _hy_text_dataset(
    queries: list[ModalQuery],
    *,
    library_dir: Path,
    cfg: Settings,
    graded_labels: dict[str, dict[str, float]] | None = None,
) -> EvaluationDataset:
    """Build the common text :class:`EvaluationDataset` with HY proxy labels.

    Each ``text`` query is labelled via :func:`proxy_labels`; only HY-labelled
    queries (the whole text subset, by construction) are kept so the table is a
    single honesty tier. ``run_retrieval_eval`` reads ``relevant_scene_ids`` /
    ``relevance`` off the ``QueryCase`` rows — i.e. the HY labels — directly.

    When ``graded_labels`` is provided, per-query relevance is taken from
    ``graded_labels[query_id]`` (with scene_id keys canonicalised via
    :func:`scene_id_key`) instead of calling :func:`proxy_labels`. Queries
    absent from ``graded_labels`` fall back to :func:`proxy_labels`. The
    label_method recorded on each row reflects the source used.

    Raises:
        EvalError: when no text query yields a usable label.
    """
    cases: list[QueryCase] = []
    for q in queries:
        if q.query_type != "text" or not q.text:
            continue

        # Prefer human grades when available for this query.
        if graded_labels is not None and q.id in graded_labels:
            raw_rel = graded_labels[q.id]
            # Canonicalise keys + keep only positive grades.
            relevance = {
                scene_id_key(k): float(v)
                for k, v in raw_rel.items()
                if float(v) > 0
            }
            if not relevance:
                # All grades non-positive → fall back to proxy so this query
                # contributes to the common set rather than being dropped.
                rel_ids, relevance, method = proxy_labels(q, library_dir=library_dir, cfg=cfg)
                if method != "HY" or not rel_ids:
                    logger.debug(
                        "ablation: skipping %s (graded all-zero + no HY, method=%s)",
                        q.id,
                        method,
                    )
                    continue
            else:
                rel_ids = tuple(relevance.keys())
                method = "GRADED"
        else:
            rel_ids, relevance, method = proxy_labels(q, library_dir=library_dir, cfg=cfg)
            if method != "HY" or not rel_ids:
                # The common set is HY-only; a text query without a usable
                # hypothesis is skipped rather than blended in under another tier.
                logger.debug("ablation: skipping %s (method=%s, ids=%s)", q.id, method, rel_ids)
                continue

        cases.append(
            QueryCase(
                id=q.id,
                text=q.text,
                relevant_scene_ids=rel_ids,
                relevance=relevance or {sid: 1.0 for sid in rel_ids},
                notes=q.notes or "",
            )
        )
    if not cases:
        raise EvalError(
            "no HY-labelled text queries available for the ablation common set "
            "(every text query must carry relevant_scene_ids / relevance)"
        )
    return EvaluationDataset(
        dataset="m3_ablation_text",
        version=1,
        queries=tuple(cases),
        source={"modality": "text", "proxy": "HY", "common_set": True},
        label_status="seed_curator_grading_pending",
        path=None,
    )


def _scope_cfg_to_film(cfg: Settings, library_dir: Path, slug: str) -> Settings:
    """Return a deep-copied cfg with ``paths.{embeddings,metadata,frames}_dir``
    pointed at one per-film library directory.

    Mirrors ``scripts/run_eval._override_film_paths`` but on a copy (the caller
    keeps the original cfg intact for other rows). ``run_retrieval_eval`` reads
    these paths to locate the per-film CLIP index + BM25 corpus.
    """
    film_dir = library_dir / slug
    scoped = copy.deepcopy(cfg)
    scoped.paths.embeddings_dir = film_dir / "embeddings"
    scoped.paths.metadata_dir = film_dir / "metadata"
    scoped.paths.frames_dir = film_dir / "frames"
    return scoped


def _scope_cfg_openclip(cfg: Settings, library_dir: Path, slug: str) -> Settings:
    """Like :func:`_scope_cfg_to_film` but switched to the OpenCLIP index.

    Points the embeddings filename/mapping at the ``.clip_openclip`` artefacts
    and the image-embedder backend at ``clip_openclip`` so ``run_retrieval_eval``
    loads the OpenCLIP keyframe matrix AND encodes the query text with the
    matching OpenCLIP text tower (dim-compatible 512-d). Both files exist on disk
    per film (the M3 SigLIP2 rollout kept them for rollback).
    """
    scoped = _scope_cfg_to_film(cfg, library_dir, slug)
    scoped.embeddings.filename = _OPENCLIP_EMB_FILENAME
    scoped.embeddings.mapping_filename = _OPENCLIP_MAP_FILENAME
    scoped.models.image_embedder = "clip_openclip"
    return scoped


def _primary_film_slug(library_dir: Path, queries: list[ModalQuery]) -> str:
    """Pick the corpus film for the text rows: the largest indexed film.

    The text HY hypotheses reference one film's scene ids; ``run_retrieval_eval``
    is single-index. We pick the film with the most CLIP-embedding rows on disk
    (Jeca Tatu at 412 scenes vs Porter at 7) so the common set scores against the
    corpus its hypotheses were written for. Falls back to the first registered
    slug if no embeddings are found.
    """
    candidates: list[tuple[int, str]] = []
    if not library_dir.exists():
        raise EvalError(f"library_dir does not exist: {library_dir}")
    for child in sorted(library_dir.iterdir()):
        if not child.is_dir():
            continue
        emb = child / "embeddings" / "keyframe_embeddings.npy"
        if emb.exists():
            try:
                n = int(np.load(emb, mmap_mode="r").shape[0])
            except Exception:  # noqa: BLE001 - unreadable index → rank last
                n = 0
            candidates.append((n, child.name))
    if not candidates:
        raise EvalError(
            f"no per-film CLIP index found under {library_dir} — cannot run the "
            "ablation text rows"
        )
    candidates.sort(reverse=True)
    return candidates[0][1]


# ─────────────────────────────────────────────────────────────────────────────
# Per-row mechanics.
# ─────────────────────────────────────────────────────────────────────────────


def _run_text_retriever_row(
    cfg: Settings,
    dataset: EvaluationDataset,
    *,
    library_dir: Path,
    slug: str,
    retriever: str,
    seed: int,
) -> dict[str, float | int]:
    """CLIP / BM25 / hybrid / multilingual row via :func:`run_retrieval_eval`.

    ``multilingual`` switches the cfg to the OpenCLIP index + text encoder; the
    others use the configured (SigLIP2) default. Returns the ``RetrievalRun``
    metrics dict.
    """
    if retriever == "multilingual":
        scoped = _scope_cfg_openclip(cfg, library_dir, slug)
        effective = "clip"
    else:
        scoped = _scope_cfg_to_film(cfg, library_dir, slug)
        effective = retriever
    run: RetrievalRun = run_retrieval_eval(
        scoped,
        dataset,
        config_path=None,
        top_k=10,
        retriever=effective,
        seed=seed,
    )
    return run.metrics


def _run_rerank_row(
    cfg: Settings,
    dataset: EvaluationDataset,
    *,
    library_dir: Path,
    slug: str,
    seed: int,
) -> dict[str, float | int]:
    """hybrid+rerank row — production ``find(mode="hybrid", rerank=True)``.

    Scores each text query against the per-film index with the production
    retrieval path so the cross-encoder (C5) reorders the hybrid top-N. The
    base is ``find``'s hybrid (NOT the harness ``hybrid`` row) — see the
    table footnote. Built from ``cinemateca.search`` only (no api import).
    """
    from cinemateca.reproducibility import seed_everything
    from cinemateca.search import Query, find

    seed_everything(seed)
    ctx = _FilmCtx.for_slug(library_dir, slug)
    scoped = _scope_cfg_to_film(cfg, library_dir, slug)

    results = []
    for case in dataset.queries:
        result = find(
            Query.of_text(case.text),
            film=ctx,
            mode="hybrid",
            top_k=20,
            rerank=True,
            rerank_model="default",
            cfg=scoped,
        )
        ranked = tuple(scene_id_key(h.scene_id) for h in result.hits)
        results.append(
            evaluate_query(
                query_id=case.id,
                text=case.text,
                relevant_scene_ids=case.relevant_scene_ids,
                ranked_scene_ids=ranked,
                relevance=case.relevance,
            )
        )
    if not results:
        raise EvalError("rerank row produced no scorable queries")
    return summarize_results(results)


def _run_fusion_row(
    cfg: Settings,
    dataset: EvaluationDataset,
    *,
    library_dir: Path,
    slug: str,
    seed: int,
) -> dict[str, float | int]:
    """fusion row — CLIP × CLAP :func:`search_fusion` per text query.

    Loads the per-film CLIP + CLAP indexes (reusing the loading
    ``cinemateca.eval.slates._slate_fusion`` does) and scores the fused ranking
    against the HY labels. A film without a CLAP index just doesn't contribute
    the audio term (graceful — Porter has no CLAP); the scored corpus is the
    text-row film (Jeca Tatu), which DOES have CLAP.
    """
    from cinemateca.models.registry import get_audio_embedder, get_image_embedder
    from cinemateca.reproducibility import seed_everything
    from cinemateca.search.audio import load_audio_index
    from cinemateca.search.fusion import FusionConfig, search_fusion

    seed_everything(seed)
    emb_dir = library_dir / slug / "embeddings"
    clip_emb_path = emb_dir / "keyframe_embeddings.npy"
    clip_map_path = emb_dir / "index_mapping.json"
    if not (clip_emb_path.exists() and clip_map_path.exists()):
        raise EvalError(f"fusion row: CLIP index missing for {slug} under {emb_dir}")

    clip_emb = np.load(clip_emb_path).astype("float32", copy=False)
    clip_mapping = _normalise_clip_mapping(json.loads(clip_map_path.read_text()))

    audio_idx = load_audio_index(library_dir / slug / "audio")
    has_clap = audio_idx is not None

    clip_embedder = get_image_embedder(cfg, device=None)
    clap_embedder = get_audio_embedder(cfg, device=None) if has_clap else _NullEncoder()
    if has_clap:
        assert audio_idx is not None
        clap_emb = audio_idx.embeddings
        clap_mapping = [{"scene_id": int(m["scene_id"])} for m in audio_idx.mapping]
    else:
        clap_emb, clap_mapping = np.zeros((0, 1), dtype="float32"), []

    results = []
    for case in dataset.queries:
        hits = search_fusion(
            clip_emb=clip_emb,
            clap_emb=clap_emb,
            clip_mapping=clip_mapping,
            clap_mapping=clap_mapping,
            query_text=case.text,
            clip_embedder=clip_embedder,
            clap_embedder=clap_embedder,
            cfg=FusionConfig(visual_weight=_FUSION_WEIGHT, k_each=50, k_final=10),
        )
        ranked = tuple(scene_id_key(h["scene_id"]) for h in hits)
        results.append(
            evaluate_query(
                query_id=case.id,
                text=case.text,
                relevant_scene_ids=case.relevant_scene_ids,
                ranked_scene_ids=ranked,
                relevance=case.relevance,
            )
        )
    if not results:
        raise EvalError("fusion row produced no scorable queries")
    return summarize_results(results)


@dataclass(frozen=True)
class _FilmCtx:
    """Minimal duck-typed ``film=`` arg for :func:`cinemateca.search.find`.

    ``find`` reads ``.slug`` / ``.embeddings_dir`` / ``.metadata_dir`` (and the
    fusion/audio paths via other callers). Built from derived paths so the
    rerank row works whether or not the slug is registry-gated — mirrors
    :class:`cinemateca.eval.slates._SlateFilmCtx`.
    """

    slug: str
    metadata_dir: Path
    embeddings_dir: Path
    audio_dir: Path

    @classmethod
    def for_slug(cls, library_dir: Path, slug: str) -> _FilmCtx:
        film_dir = library_dir / slug
        return cls(
            slug=slug,
            metadata_dir=film_dir / "metadata",
            embeddings_dir=film_dir / "embeddings",
            audio_dir=film_dir / "audio",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public entrypoint.
# ─────────────────────────────────────────────────────────────────────────────


def run_ablation(
    cfg: Settings,
    *,
    library_dir: Path,
    queries: list[ModalQuery],
    configs: tuple[AblationRowConfig, ...] = DEFAULT_ABLATION_CONFIGS,
    seed: int = 0,
    graded_labels: dict[str, dict[str, float]] | None = None,
    validated_label: str | None = None,
) -> AblationTable:
    """Run each row's retriever on the common text query set with HY labels.

    Builds the common HY-labelled text :class:`EvaluationDataset` once, picks
    the corpus film (largest indexed — Jeca Tatu on the demo library), then
    fills each row's metrics by dispatching on ``AblationRowConfig.retriever``.
    A row with ``pending_reason`` set is skipped (rendered ``pending (...)``);
    a row whose computation raises is also marked ``pending`` (with the
    exception class as the reason) rather than aborting the whole table or
    fabricating a number.

    Args:
        cfg: loaded :class:`~cinemateca.config.Settings`. Copied per row before
            any per-film / backend-switch mutation.
        library_dir: the library root (``data/library``).
        queries: parsed :class:`ModalQuery` list (only the text subset is used).
        configs: the rows to compute (default :data:`DEFAULT_ABLATION_CONFIGS`).
        seed: PRNG seed forwarded to every row for reproducibility.
        graded_labels: optional per-query relevance from human grades.
            When provided, ``{query_id: {scene_id: float_grade}}`` maps take
            precedence over :func:`proxy_labels` for queries present in the
            dict (positive grades only). Queries absent fall back to proxy.
            Without ``--grades``, behavior is byte-for-byte unchanged (still proxy).
        validated_label: when provided, the :class:`AblationTable` banner flips
            from the proxy wording to this string (e.g. ``"human-validated (run
            <id>, n=<N> grades)"``). Only meaningful when ``graded_labels`` is
            set; ignored otherwise.

    Returns:
        An :class:`AblationTable` ready to ``to_markdown()``.
    """
    dataset = _hy_text_dataset(
        queries, library_dir=library_dir, cfg=cfg, graded_labels=graded_labels
    )
    slug = _primary_film_slug(library_dir, queries)
    corpus = _corpus_description(library_dir, slug, dataset)

    rows: list[tuple[AblationRowConfig, dict[str, float | int] | None]] = []
    footnotes: dict[str, str] = {}
    for row_cfg in configs:
        if row_cfg.name in _ROW_FOOTNOTES:
            footnotes[row_cfg.name] = _ROW_FOOTNOTES[row_cfg.name]

        if row_cfg.pending_reason:
            rows.append((row_cfg, None))
            continue

        try:
            metrics = _dispatch_row(
                cfg,
                dataset,
                row_cfg,
                library_dir=library_dir,
                slug=slug,
                seed=seed,
            )
            rows.append((row_cfg, metrics))
        except Exception as exc:  # noqa: BLE001 - a failed row → honest pending
            reason = type(exc).__name__
            logger.warning("ablation row %r failed → pending (%s): %s", row_cfg.name, reason, exc)
            # Re-tag the row so its rendered reason reflects the failure cause,
            # never a fabricated number.
            failed = AblationRowConfig(
                name=row_cfg.name,
                retriever=row_cfg.retriever,
                proxy=row_cfg.proxy,
                rerank=row_cfg.rerank,
                pending_reason=reason,
            )
            rows.append((failed, None))

    return AblationTable(
        rows=rows,
        corpus=corpus,
        common_query_set=f"{len(dataset.queries)} text queries (m3_full)",
        footnotes=footnotes,
        validated_label=validated_label,
    )


def _dispatch_row(
    cfg: Settings,
    dataset: EvaluationDataset,
    row_cfg: AblationRowConfig,
    *,
    library_dir: Path,
    slug: str,
    seed: int,
) -> dict[str, float | int]:
    """Route one row config to its mechanics. Raises on an unknown retriever."""
    retriever = row_cfg.retriever
    if row_cfg.rerank:
        return _run_rerank_row(cfg, dataset, library_dir=library_dir, slug=slug, seed=seed)
    if retriever in ("clip", "bm25", "hybrid", "multilingual"):
        return _run_text_retriever_row(
            cfg, dataset, library_dir=library_dir, slug=slug, retriever=retriever, seed=seed
        )
    if retriever == "fusion":
        return _run_fusion_row(cfg, dataset, library_dir=library_dir, slug=slug, seed=seed)
    raise EvalError(f"unknown ablation retriever {retriever!r}")


def _corpus_description(library_dir: Path, slug: str, dataset: EvaluationDataset) -> str:
    """Human banner string: ``<title> (<year>) — <N> scenes, <Q> queries``."""
    title, year, n_scenes = slug, None, None
    try:
        from cinemateca.library import Library

        film = Library(library_dir).get_film(slug)
        title, year = film.title, film.year
    except Exception:  # noqa: BLE001 - degrade to slug
        pass
    try:
        emb = library_dir / slug / "embeddings" / "keyframe_embeddings.npy"
        if emb.exists():
            n_scenes = int(np.load(emb, mmap_mode="r").shape[0])
    except Exception:  # noqa: BLE001
        pass
    bits = [title]
    if year:
        bits.append(f"({year})")
    head = " ".join(bits)
    tail = []
    if n_scenes is not None:
        # The CLIP matrix is keyframe-level (several keyframes per scene); the
        # eval ranks scenes after dedup. Label it honestly as keyframes.
        tail.append(f"{n_scenes} keyframes indexed")
    tail.append(f"{len(dataset.queries)} text queries")
    return f"{head} — " + ", ".join(tail)


__all__ = [
    "AblationRowConfig",
    "AblationTable",
    "DEFAULT_ABLATION_CONFIGS",
    "DEFAULT_ABLATION_CONFIGS_NO_RERANK",
    "run_ablation",
]
