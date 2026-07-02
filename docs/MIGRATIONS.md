# Embedding migrations

## 2026-05-25 — Jeca Tatu re-embedded with SigLIP2-multilingual

- Backend: `google/siglip2-large-patch16-256` (dim 1024, multilingual)
- Previous backend: OpenClip ViT-B/32 (dim 512, English-only)
- Wall-clock: 69.7s on GPU (embeddings step only, consumer NVIDIA GPU).
- Output: 1236 keyframe vectors × 1024 dim, all L2-normalised (norms = 1.0).

### Plan-vs-reality notes

The pre-flight plan for this rollout specified
`google/siglip-large-patch16-256-multilingual`, which doesn't exist on
HuggingFace. SigLIP2-large-256 is the closest real "large + multilingual"
checkpoint (Feb 2025; the original SigLIP only ships multilingual at base).
The config knob `embeddings.model_id` carries the real id.

The registry route was added on the pipeline side (write path), but the
`kuaa.search` package still hardcoded `OpenClipEmbedder()` at two query-time
callsites — `kuaa/search/cache.py:_load_and_validate` (the per-film
`SearchIndex.embedder` field) and `kuaa/search/aggregate.py:_get_embedder`
(the aggregate text encoder). Once `cfg.models.image_embedder` flipped to
`siglip_multilingual` and the on-disk Jeca Tatu index was regenerated at
1024 dim, the live encoder still returned 512-dim vectors → matmul
size-mismatch `ValueError` → HTTP 500 on every Brazilian-Portuguese search.

Fix:

* `kuaa/search/aggregate.py:_get_embedder` — switched to
  `get_image_embedder(cfg)` (the function already had `cfg` in scope).
* `kuaa/search/cache.py:_load_and_validate` — switched to
  `get_image_embedder(get_config())` via a lazy `api.deps.get_config`
  import. This is a layering wart (the `kuaa.search` package reaching into
  `api.deps`) tracked as a follow-up: thread `cfg` through `load_index` /
  `_load_and_validate` so the search module no longer crosses the layer
  boundary.
* Test fixture (`tests/conftest.py:tmp_config`) pins
  `cfg.models.image_embedder = "clip_openclip"` for hermetic tests so
  the existing `OpenClipEmbedder` monkeypatches keep working through
  the registry. SigLIP-specific tests construct
  `SiglipMultilingualEmbedder` directly and bypass `tmp_config`, so
  the pin doesn't suppress real SigLIP coverage.

### Hand-check results (PT-BR queries, Jeca Tatu, top-5)

All three queries return HTTP 200 with five distinct scenes. SigLIP2
uses sigmoid loss (not contrastive), so absolute cosine magnitudes are
typically lower than CLIP — relative ranking is what matters.

```
=== cachorro correndo (top_score=0.055) ===
scene_ids: 1, 412, 331, 118, 29

=== homem com chapéu (top_score=0.075) ===
scene_ids: 1, 412, 331, 118, 99

=== casa de pau-a-pique (top_score=0.059) ===
scene_ids: 1, 412, 331, 99, 118
```

The first three scenes overlap across queries; positions 4–5 diverge.
Whether this overlap is the "all-paths-lead-to-scene-1" pathology of
an under-tuned encoder or a genuine semantic clustering in Jeca Tatu's
visual vocabulary is an open question — a library-wide acceptance pass is
the place to grade quality against the OpenClip baseline (CLIP backup
preserved on disk, see Rollback).

### Known limitations right after the initial re-embed

* **Aggregate (cross-film) search still 500s** when more than one
  registered film has CLIP-dim (512) embeddings and the SigLIP2 query
  vector is 1024-dim. The walk in `aggregate_search` does
  `idx.embeddings @ text_vec` per film and the second film
  (`edwin_porter-the_great_train_robbery_1903`) is still on CLIP. The
  per-film path (with `?film=jeca_tatu`) is unaffected.

  This is the gating condition for the library-wide rollout below. Until
  that rollout runs, the aggregate route should be exercised only when
  every registered film shares the same embedder backend.

### Rollback procedure

If SigLIP2 retrieval quality is worse than CLIP for institutional
queries, the previous CLIP artefacts are preserved on disk:

1. `git checkout config/default.yaml` — reverts
   `models.image_embedder` to `clip_openclip` and `embeddings.model_id`
   to the SigLIP plan id (no-op under the OpenClip backend).
2. `cd data/library/jeca_tatu/embeddings/`
3. `mv keyframe_embeddings.clip_openclip.npy keyframe_embeddings.npy`
4. `mv index_mapping.clip_openclip.json index_mapping.json`
5. Restart `uv run kuaa serve`.

The backup files are gitignored (under `data/library/`).

### Library-wide rollout — 2026-05-25

Per the known cross-film aggregate limitation above (500s on mixed dims),
the rollout to the remaining film completed the same day:

- `edwin_porter-the_great_train_robbery_1903` — re-embedded with SigLIP2.
  - Shape: (21, 1024) — 21 keyframes (short film).
  - Wall-clock: 7.4s on GPU.
  - CLIP backup preserved as `.clip_openclip.npy` in the same dir.

Library is now uniformly SigLIP2-large-256. Cross-film aggregate retrieval restored:

- HTTP 200 on `/api/search?q=...&top_k=N` (no `?film=` filter).
- Hits returned from both films. Verified via server logs from the
  acceptance run: `aggregate_search: films=2 ... clip_n=5 ... clip_n=40`
  for `q='train robbery'` (Edwin Porter contributes 5 candidates,
  Jeca Tatu 40, merged into a single ranked list — previously 500).
- Top-10 results are dominated by Jeca Tatu on most queries: it has
  1236 keyframes vs Edwin Porter's 21, and SigLIP2 produces higher
  absolute scores on Jeca Tatu's colour/curated content than on the
  1903 silent print. This is a calibration property of the corpus
  asymmetry, not a defect of the aggregate path. Per-film queries
  on either slug still return that film's own scenes correctly.

Per-film rollback procedure is the same as documented above — flip
`models.image_embedder` back to `clip_openclip` AND restore the per-film
`.clip_openclip.npy` files. A partial rollback (only one film) re-introduces
the dim-mismatch breakage; either roll back all films or none.

## Current state (both layouts coexist)

As of this writing, `config/default.yaml` sets `models.image_embedder:
siglip_multilingual` as the shipped default — the migration above is
complete, not just proposed. On disk, both the registry-backed per-film
layout (`data/library/<slug>/...`) and the legacy flat layout
(`data/{frames,metadata,embeddings,raw}`) currently coexist, matching the
dual-path support described in `docs/ARCHITECTURE.md`. Anything that reads
embeddings directly (rather than through `kuaa.models.registry`) should
still expect either 512-dim (OpenCLIP/M-CLIP) or 1024-dim (SigLIP 2)
vectors depending on which layout and backend produced them, and should
not assume a single dimension library-wide without checking
`index_mapping.json` provenance first.
