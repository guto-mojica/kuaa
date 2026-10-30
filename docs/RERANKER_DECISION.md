# Cross-encoder reranker — disabled by default (tracked debt)

**Status:** Disabled by default (`retrieval.reranker.enabled: false`) as of
2026-05-31, and still disabled by default in the current 0.8.0rc1 tree
(confirmed against the live `config/default.yaml`). The reranker stays in the
tree as an opt-in (`?reranker_enabled=true` per request, or pin the config to
`true`/`auto`); it is *not* removed. **This is a deliberate interim state,
not the end state** — see "Why this can't stand for the portfolio" and "The
fork" below.

---

## Decision

Ship with the cross-encoder reranker (`BAAI/bge-reranker-v2-m3`, the `rerank`
verb in `kuaa.search.rerank`) **off by default**. The first-stage retriever
(SigLIP2 visual ⊕ BM25 hybrid via RRF) is the production ranking.

## Why it's off

Three reasons, in order of weight:

1. **Its true effect was unmeasured at decision time.** An ablation row
   (`hybrid+rerank` nDCG 0.388 vs `hybrid` 0.447) was produced by the **core
   `find(rerank=True)` path, which at the time reranked empty descriptions**:
   the loaded keyframe index carried only `filepath` + `scene_id`
   (`openclip.py`), and `_df_to_result` defaulted the missing `description`
   column to `""` (`_dispatch.py`), so the cross-encoder scored
   `[query, ""]` pairs. That number measured rerank-on-nothing, **not**
   rerank-on-captions — it was an artifact, not evidence. The live HTTP
   path *did* populate descriptions (enrichment reads
   `scene_descriptions.json`), but that path was never ablated at the time.
   **Update since the original decision:** this specific measurement gap has
   since been closed. `kuaa.search._dispatch.find()` now calls a
   `_attach_descriptions()` step that fills empty `Hit.description` from the
   film's `scene_descriptions.json` before reranking, so the core path no
   longer reranks blank captions. That fixes the *prerequisite* for an
   honest re-measurement (see Acceptance criteria below); it does not, by
   itself, re-run the ablation on curator-graded data, so the reranker
   remains off by default pending that follow-up.
2. **The design is suspect even with real captions** (the live path). A text-only
   cross-encoder over short, English-leaning Moondream captions, scoring a modality
   the first stage already covered better — see "Root cause" below. The likely
   live-path effect is neutral-to-negative, but that too remains unmeasured on
   curator-graded data.
3. **Cost.** The ~2.4 GB cross-encoder is multi-second per query on CPU (the
   reason the config exposes an `auto` GPU/CPU split). Shipping a multi-second
   component of unknown value default-on is indefensible.

### Root cause — why, not just that

A second-stage reranker earns its place only if it (1) sees a signal the first stage
didn't and (2) is *more precise* on that signal. The current setup violates **both**:

1. **It scores the wrong modality.** `rerank()` feeds the cross-encoder
   `(query, hit.description)` — the Moondream *caption*, a lossy text proxy of the
   image. The first stage (SigLIP2) scored the actual **pixels**. So the reranker
   isn't adding signal; it substitutes a weaker one. Caption/label coverage is
   already a known ceiling on ranking quality — the reranker reranks on the
   weakest part of the pipeline. *(Evidence: `h.description` in
   `kuaa/search/rerank.py`.)*
2. **It replaces instead of refines.** `rerank()` sorts purely by `rerank_score`,
   discarding the first-stage rank. A confident rank-1 visual match with a thin
   caption gets buried under a cross-encoder logit. Worse, the two scores live on
   incomparable scales (CLIP cosine ≈ [0,1] vs. unbounded/negative cross-encoder
   logits), so the hard sort treats a non-comparable number as authoritative.

On short, generic film captions — the exact regime here — a general-domain
cross-encoder is at its noisiest, so substitution + hard-replace degrades a calibrated
visual signal. That is the whole story behind the negative delta.

## Why this can't stand for the portfolio

This project is an applied-ML / retrieval portfolio piece. "I bolted on a reranker"
is weak; "I added one then turned it off" is weaker. Leaving a measured-negative,
on-the-shelf component in the codebase reads as an unfinished idea. The interim
disable buys a clean launch, **but the reranker must be fixed properly or
removed before the project is presented as finished retrieval work.**

## The fork (what "fixed properly" means)

Two ways to make the two stages cooperate, plus the exit option:

- **A — Fuse, don't replace (recommended near-term).** RRF-fuse the reranker's
  ranking with the first-stage ranking instead of overriding it — the same
  rank-fusion primitive already used for CLIP⊕BM25 (`kuaa.retrieval`). It is
  calibration-free (sidesteps the scale mismatch), bounds the reranker to a *vote*
  that can nudge near-ties but never bury a confident visual hit, and exposes a
  weight knob for the ablation to tune. Realistic outcome: moves rerank from
  "hurts" to "neutral-to-slightly-helps" — it does **not** beat the caption-quality
  ceiling. Cheap, low-risk, and the better narrative ("diagnosed the antagonism,
  fixed it with rank fusion, validated by ablation").
- **B — Cross-modal reranker (the real ceiling-raiser).** A VLM-as-judge that scores
  the **keyframe image** + query, not the caption — restores the missing visual
  signal (condition #1). Heavier (VLM inference per candidate; gate to small
  candidate sets) but the architecturally honest answer for a *visual* archive.
- **C — Remove entirely.** If neither A nor B clears the acceptance bar, delete the
  `rerank` verb, the config block, and the UI affordance rather than ship dead capability.

The **first-stage pool widening** (fetch `max(top_k+offset, top_k_in)` when
rerank is on; see `api/services/_search_render.py`) is the prerequisite for A
and B — without depth in the candidate pool the reranker can only reorder the
visible page. It is retained so the fix path is ready to build on.

## Acceptance criteria to re-enable by default

Re-flip `retrieval.reranker.enabled` to `auto`/`true` only when **all** hold:

0. **Prerequisite — fix the measurement.** *(Status: done.)* The core
   `find(rerank=True)` path now loads real scene descriptions (by `scene_id`
   from `scene_descriptions.json` via `_attach_descriptions()`) instead of
   defaulting to `""`, so a fresh ablation would actually feed the
   cross-encoder real captions. Every ablation number produced *before* this
   fix landed should still be treated as an artifact.
1. An ablation on **curator-graded** data (not proxy) shows rerank-on ≥ hybrid
   baseline on nDCG@5 **and** does not regress PT-language queries. *(Status:
   measured for the current text-only hard-replace; see "Measured on graded
   data" below. nDCG@5 was not computed.)*
2. Per-query latency stays within budget on the target profile (the `auto`
   GPU/CPU split exists precisely because the ~2.4 GB cross-encoder is
   multi-second on CPU).
3. The chosen approach (A or B) is the one measured — not the current text-only
   hard-replace.

## Measured on graded data

The fixed core path was run on the graded `corpus01` pool (56 queries, one
grader, 2436 grades; `docs/EVALUATION_RESULTS.md`). This is the current
design: the cross-encoder reads the caption and hard-replaces the first-stage
order. It is neither fork A nor fork B.

| | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| hybrid | 0.178 | 0.344 | 0.920 | 0.677 |
| hybrid_rerank | 0.188 | 0.355 | 0.932 | 0.676 |

- **No measurable effect.** nDCG@10 differs by 0.001; the recall and MRR gains
  (about +0.01) are inside the noise of 56 queries and one grader.
- **No PT regression on nDCG@10.** PT 0.672 against 0.666 for `hybrid`.
- **Large per-query swings that cancel.** 18 queries improve, 23 worsen and 15
  tie, by up to ±0.6. It is worse on short queries (0.654 against 0.724) and
  better on mid-length ones (0.666 against 0.610); those groups hold 20
  queries each.
- **Pooled evaluation understates it.** A candidate the reranker promotes that
  no pooled variant proposed was never shown to a grader and counts as
  irrelevant.

Criterion 1 is therefore met only in the weak sense that it does not regress.
Nothing here is a reason to enable it, so it stays off. Fork A is the cheap
next measurement; fork B overlaps with testing a different describer.

## What ships

- `config/default.yaml` → `retrieval.reranker.enabled: false`.
- The `rerank` verb, the typed `SearchResult` rerank boundary, the per-request
  `?reranker_enabled=` override, and the first-stage widening all remain in place.
- The Buscar UI rerank toggle remains; it seeds *off* and is an explicit opt-in.

## Pointers

- Measured deltas: the ablation row referenced above (§ "Why it's off", point 1)
  is the current evidence trail, and no standalone write-up of that ablation or
  its failure analysis exists in this repository. The published table is
  `docs/EVALUATION_RESULTS.md`, whose `hybrid_rerank` row is the graded
  measurement summarised under "Measured on graded data".
- Forward plan: deepening the first-stage candidate pool ahead of fork A or B
  has no roadmap entry of its own. What it would build on is already in the
  tree — `retrieval.reranker.top_k_in` and the widening in
  `api/services/_search_render.py` — so the prerequisite is the measurement,
  not the mechanism.
- Code: `kuaa/search/rerank.py` (the `rerank` verb), `api/services/_search_rerank.py`
  (config-aware wrapper + typed boundary), `api/services/_search_render.py`
  (first-stage widening), `kuaa/search/_dispatch.py` (`_attach_descriptions()`,
  the core-path description fix referenced in point 0 above).
