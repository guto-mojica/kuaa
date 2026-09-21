"""Proxy-first retriever-variant ablation table.

This module produces a retriever-variant ablation table that is **publishable
with zero human grades**. Every row is scored on a *common query set* with the
*same* proxy labels (apples-to-apples), the proxy signal is named in a ``Proxy``
column, and any backend that is not wired renders a literal ``pending (...)``
cell — never a fabricated or zero number.

Design
------
The common query set must carry the maintainer's pre-curator hypothesis
(``relevant_scene_ids`` / ``relevance``) on every text query, so
:func:`kuaa.eval.proxy.proxy_labels` returns ``"HY"`` for all of them and
the whole table is **one honesty tier** — no tautological pseudo-relevance, no
structurally-zero rhyme row blended into the average.

Rows come from :data:`kuaa.eval.registry.RETRIEVER_REGISTRY`, which is also
what builds the graded pool. That is deliberate: grades persist the pool's
spelling of a variant name, so a table that spelled the same rows ``CLIP`` /
``hybrid-metadata`` / ``hybrid+rerank`` could not be joined to the grades meant
to score it — and the join failed by finding nothing, not by raising.

======================  =======================================================  =====
row                     how                                                      proxy
======================  =======================================================  =====
``clip``                :func:`run_retrieval_eval` (SigLIP2 default index)       HY
``bm25``                :func:`run_retrieval_eval` ``retriever="bm25"``          HY
``hybrid``              :func:`run_retrieval_eval` ``retriever="hybrid"`` —      HY
                        the SHIPPED 3-way fusion (CLIP + BM25 + metadata)
``hybrid_no_metadata``  same, ``metadata_w=0.0`` — isolates the metadata leg     HY
``hybrid_rerank``       production ``find(mode="hybrid", rerank=...)``           HY
======================  =======================================================  =====

The reranker only scores **text** queries (it reads ``query.text``), which is
the common set, so the rerank delta is well-defined. The rerank row uses the
production :func:`kuaa.search.find` for *both* its hybrid base
(``rerank=False``) and its reranked variant (``rerank=True``); the plain
``hybrid`` row above is the harness's own ``run_retrieval_eval`` path. They are
two different hybrid implementations, so the table footnote states the rerank
delta is measured on ``find``'s hybrid, not on the harness hybrid row.

A ``multilingual`` row (OpenCLIP baseline vs the SigLIP2 default) was
removed 2026-08-04: it assumed on-disk ``.clip_openclip`` artefacts that no
commit in this repo's history ever produced for any film, so the row only
ever rendered ``pending (EvalError)``. Re-add it if/when a real OpenCLIP
index is generated and the comparison is actually wanted.

Layering: this is core (``kuaa.*``); it MUST NOT import ``api.*``
(import-linter). The reranker (:func:`kuaa.search.rerank.rerank` via
``find``) is ``kuaa``-side.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kuaa.config import Settings
from kuaa.errors import EvalError
from kuaa.eval.datasets import EvaluationDataset, QueryCase
from kuaa.eval.metrics import evaluate_query, summarize_results
from kuaa.eval.proxy import proxy_labels
from kuaa.eval.registry import RETRIEVER_REGISTRY, RetrieverVariant
from kuaa.eval.retrieval import RetrievalRun, run_retrieval_eval
from kuaa.eval.slates import ModalQuery
from kuaa.scene_ids import scene_id_key

logger = logging.getLogger(__name__)

# Metric keys rendered as table columns, in display order. Maps the
# ``summarize_results`` keys to their published column headers.
_METRIC_COLUMNS: tuple[tuple[str, str], ...] = (
    ("recall_at_5", "Recall@5"),
    ("recall_at_10", "Recall@10"),
    ("mrr", "MRR"),
    ("ndcg_at_10", "nDCG@10"),
)


@dataclass(frozen=True)
class AblationRowConfig:
    """One row of the ablation table.

    Attributes:
        name: row label; the registry variant name, which is also the
            spelling grades persist (e.g. ``"hybrid_rerank"``).
        retriever: which retriever mechanism to run — ``"clip" | "bm25" | "hybrid"``.
        proxy: the proxy signal used for the row's labels — ``"KI" | "PR" |
            "HY"`` (the whole launch table is ``"HY"``; the field exists so a
            future mixed table can segregate tiers).
        rerank: when ``True`` the row applies the cross-encoder reranker on top of a
            ``find``-based hybrid base (only meaningful with
            ``retriever == "hybrid"``).
        metadata_w: hybrid-only — the exact-lexical metadata list's fusion
            share, forwarded to :func:`run_retrieval_eval`. ``None`` (default)
            resolves ``cfg.search.hybrid_metadata_w`` so the ``hybrid`` row
            measures the shipped 3-way fusion; ``0.0`` disables the leg (the
            signal-level ablation arm).
        pending_reason: when set, the row is rendered ``pending (<reason>)`` and
            :func:`run_ablation` does not attempt to compute it. Used for a row
            whose backend is not wired — the reason is printed verbatim into
            the published table, so make it a phrase a reader can act on
            (e.g. ``"rerank off"``), not an internal ticket code.
    """

    name: str
    retriever: str
    proxy: str = "HY"
    rerank: bool = False
    metadata_w: float | None = None
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
            # No KI/PR/HY list here: those are the proxy tiers, and a graded
            # table has no proxy tier to explain. Leaving the heading in place
            # ended the banner on "Proxy signals:" followed by the table.
            return [
                f"**Human-validated methodology.** Labels are {self.validated_label}. "
                "Every row is scored on the same query set against the same human "
                "grades, so the comparison is apples-to-apples.",
                "",
                f"**Corpus.** {self.corpus or 'demo library'}.",
                f"**Common query set.** {self.common_query_set or 'text queries'}.",
                "",
            ]
        return [
            "**Proxy methodology.** These are **proxy metrics**, not human-graded "
            "ground truth — they upgrade to human-validated numbers once curator "
            "grades are recorded in the admin `/eval` grading UI and passed back "
            "via `run_ablation.py --grades`. Every row below is scored on a common query "
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
        label_col = "Labels" if self.validated_label else "Proxy"
        headers = ["Retriever", label_col, *[label for _key, label in _METRIC_COLUMNS]]
        sep = ["---", "---", *["---:" for _ in _METRIC_COLUMNS]]
        lines = self._banner()
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(sep) + " |")

        for row_cfg, metrics in self.rows:
            # A graded row's label source is the grade log, not the row's proxy
            # tier — rendering "HY" beside a human-graded number claims the
            # wrong provenance for it.
            cells = [row_cfg.name, "graded" if self.validated_label else row_cfg.proxy]
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


def _row_from_variant(
    variant: RetrieverVariant, *, pending_reason: str | None = None
) -> AblationRowConfig:
    """One table row per registry variant, sharing the variant's name.

    The name matters more than it looks. Grades persist the *pool* spelling of
    a variant (``hybrid_no_metadata``); this table used to spell the same five
    rows ``CLIP`` / ``hybrid-metadata`` / ``hybrid+rerank``. Joining a graded
    pool to the table meant to score it found nothing under those names, and
    nothing raised — the join just came up empty. One name-space, derived, so
    the drift cannot come back on the next variant.
    """
    return AblationRowConfig(
        name=variant.name,
        retriever=variant.mode,
        proxy="HY",
        rerank=variant.rerank,
        metadata_w=variant.metadata_w,
        pending_reason=pending_reason,
    )


# Full set (rerank row REAL — uses production find ± the reranker). The ``hybrid``
# row measures the SHIPPED 3-way fusion (metadata_w=None → cfg default);
# ``hybrid_no_metadata`` is identical except the metadata leg is off, so the
# delta between the two isolates the signal's contribution.
DEFAULT_ABLATION_CONFIGS: tuple[AblationRowConfig, ...] = tuple(
    _row_from_variant(v) for v in RETRIEVER_REGISTRY.values()
)

# No-rerank variant — the rerank row is left pending so the table is produced
# without paying the cross-encoder cost (and the committed --no-rerank doc is
# honest about which rows are real).
DEFAULT_ABLATION_CONFIGS_NO_RERANK: tuple[AblationRowConfig, ...] = tuple(
    _row_from_variant(v, pending_reason="rerank off" if v.rerank else None)
    for v in RETRIEVER_REGISTRY.values()
)

# Footnotes attached to the rendered table when the matching row is present.
# Sourced from the registry so a variant's explanation lives with its definition.
_ROW_FOOTNOTES: dict[str, str] = {
    v.name: v.footnote for v in RETRIEVER_REGISTRY.values() if v.footnote
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
    cross_film: bool = False,
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
            relevance = {scene_id_key(k): float(v) for k, v in raw_rel.items() if float(v) > 0}
            if not relevance and cross_film:
                # Graded, but nothing in the pool was relevant. There is no
                # cross-film proxy to fall back to, and a query with no
                # relevant scene scores 0 for every row by construction.
                logger.info("ablation: skipping %s — graded with no positive label", q.id)
                continue
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
        elif cross_film:
            # Proxy labels are bare scene ordinals scoped to one film; a
            # cross-film row ranks "<slug>/<scene_id>". Blending the two would
            # score this query 0 against every row and pull the whole table
            # down by an equal amount — invisible in the output, since a
            # uniform drag still looks like a comparison.
            logger.info("ablation: skipping %s — no human grades, and the run is cross-film", q.id)
            continue
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
            f"no per-film CLIP index found under {library_dir} — cannot run the ablation text rows"
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
    metadata_w: float | None = None,
    seed: int,
) -> dict[str, float | int]:
    """CLIP / BM25 / hybrid row via :func:`run_retrieval_eval`.

    Uses the configured (SigLIP2) default index. ``metadata_w`` (hybrid rows
    only) forwards the row's metadata-leg share — ``None`` = shipped default,
    ``0.0`` = the signal-level ablation arm. Returns the ``RetrievalRun``
    metrics dict.
    """
    scoped = _scope_cfg_to_film(cfg, library_dir, slug)
    run: RetrievalRun = run_retrieval_eval(
        scoped,
        dataset,
        config_path=None,
        top_k=10,
        retriever=retriever,
        metadata_w=metadata_w,
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
    """hybrid_rerank row — production ``find(mode="hybrid", rerank=True)``.

    Scores each text query against the per-film index with the production
    retrieval path so the cross-encoder reranker reorders the hybrid top-N. The
    base is ``find``'s hybrid (NOT the harness ``hybrid`` row) — see the
    table footnote. Built from ``kuaa.search`` only (no api import).
    """
    from kuaa.reproducibility import seed_everything
    from kuaa.search import Query, find

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


def _run_variant_row_global(
    cfg: Settings,
    dataset: EvaluationDataset,
    *,
    library_dir: Path,
    variant: RetrieverVariant,
    seed: int,
    top_k: int = 10,
) -> dict[str, float | int]:
    """One row over the library-wide ranking, keyed ``<slug>/<scene_id>``.

    Used whenever the labels are human grades. A grade's key is
    ``(query_id, "<film_slug>/<scene_id>")`` because ``scene_id`` alone is
    unique only within a film and a pool spans the library — so the ranking
    scored against it has to span the library too, and carry the same key.
    The single-film rows below cannot: they scope the config to one slug and
    rank bare ordinals, which joins to nothing and reads as 0.000 across the
    whole table.

    Retrieval comes from :func:`kuaa.eval.slates.rank_variant_global` — the
    same call that built the pool — so the table cannot score a path the
    grades do not cover.
    """
    from kuaa.eval.slates import _film_meta_loader, rank_variant_global
    from kuaa.reproducibility import seed_everything
    from kuaa.search import Query

    seed_everything(seed)
    load_meta = _film_meta_loader(cfg, library_dir)

    results = []
    for case in dataset.queries:
        rows = rank_variant_global(
            q=Query.of_text(case.text),
            cfg=cfg,
            library_dir=library_dir,
            k=top_k,
            variant=variant,
            load_meta=load_meta,
        )
        ranked = tuple(f"{r['film_slug']}/{scene_id_key(r['scene_id'])}" for r in rows)
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
        raise EvalError("cross-film row produced no scorable queries")
    return summarize_results(results)


@dataclass(frozen=True)
class _FilmCtx:
    """Minimal duck-typed ``film=`` arg for :func:`kuaa.search.find`.

    ``find`` reads ``.slug`` / ``.embeddings_dir`` / ``.metadata_dir``. Built
    from derived paths so the rerank row works whether or not the slug is
    registry-gated — mirrors :class:`kuaa.eval.slates._SlateFilmCtx`.
    """

    slug: str
    metadata_dir: Path
    embeddings_dir: Path

    @classmethod
    def for_slug(cls, library_dir: Path, slug: str) -> _FilmCtx:
        film_dir = library_dir / slug
        return cls(
            slug=slug,
            metadata_dir=film_dir / "metadata",
            embeddings_dir=film_dir / "embeddings",
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
        cfg: loaded :class:`~kuaa.config.Settings`. Copied per row before
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
    # Human grades are keyed "<slug>/<scene_id>" across the library, so they
    # can only score a ranking that spans it. Proxy labels are bare ordinals
    # from one film's query file, so they can only score the single-film rows.
    # The label space, not a flag, decides which retrieval the table runs.
    cross_film = graded_labels is not None
    dataset = _hy_text_dataset(
        queries,
        library_dir=library_dir,
        cfg=cfg,
        graded_labels=graded_labels,
        cross_film=cross_film,
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
                cross_film=cross_film,
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
                metadata_w=row_cfg.metadata_w,
                pending_reason=reason,
            )
            rows.append((failed, None))

    return AblationTable(
        rows=rows,
        corpus=corpus,
        common_query_set=f"{len(dataset.queries)} text queries",
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
    cross_film: bool = False,
) -> dict[str, float | int]:
    """Route one row config to its mechanics. Raises on an unknown retriever."""
    if cross_film:
        variant = RETRIEVER_REGISTRY.get(row_cfg.name)
        if variant is None:
            raise EvalError(
                f"row {row_cfg.name!r} is not in the retriever registry — a cross-film row "
                f"is run from the registry variant so it matches the pool the grades came from"
            )
        return _run_variant_row_global(
            cfg, dataset, library_dir=library_dir, variant=variant, seed=seed
        )
    retriever = row_cfg.retriever
    if row_cfg.rerank:
        return _run_rerank_row(cfg, dataset, library_dir=library_dir, slug=slug, seed=seed)
    if retriever in ("clip", "bm25", "hybrid"):
        return _run_text_retriever_row(
            cfg,
            dataset,
            library_dir=library_dir,
            slug=slug,
            retriever=retriever,
            metadata_w=row_cfg.metadata_w,
            seed=seed,
        )
    raise EvalError(f"unknown ablation retriever {retriever!r}")


def _corpus_description(library_dir: Path, slug: str, dataset: EvaluationDataset) -> str:
    """Human banner string: ``<title> (<year>) — <N> scenes, <Q> queries``."""
    title, year, n_scenes = slug, None, None
    try:
        from kuaa.library import Library

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
