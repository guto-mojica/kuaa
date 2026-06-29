# Case study: offline multimodal search for archival video

## Summary

KUAA is a local-first applied AI workbench for turning private
video collections into searchable, human-reviewable scene catalogs. The flagship
domain is film archives: historical footage, sparse metadata, digitized material
with variable quality, and institutional constraints around privacy and
provenance.

The project demonstrates a full applied-ML workflow rather than a single model
demo:

- ingest a video,
- detect scenes and extract representative keyframes,
- generate local visual metadata and scene descriptions,
- index scenes for text and image search,
- let humans correct tags and improve catalog quality,
- evaluate retrieval quality with a fixed query set,
- adapt prompts and export schemas through domain packs,
- export a structured catalog,
- write run manifests so outputs can be traced to inputs, config, and models.

The public demo uses Library of Congress footage, not private institutional data.
The production design remains intentionally local, single-machine, and
single-user for the current public release path.

## Problem

Archives and cinematheques often hold large digitized video collections with
minimal scene-level metadata. A film may have a title, date, creator, and broad
description, while the moments that researchers actually search for are buried
inside the footage: people at a train station, indoor conversations, crowd
scenes, signage, vehicles, landscapes, or historically relevant objects.

Manual scene cataloging is expensive. Fully automated cataloging is not reliable
enough to replace curatorial review. The useful product shape is a first-pass AI
system that makes visual material discoverable quickly, then keeps human review
in the loop.

## Constraints

The project was designed around constraints that are common in cultural heritage
and private visual collections:

- **Privacy**: videos, keyframes, search queries, annotations, and embeddings
  should stay on the local machine after dependencies and model weights are
  installed.
- **Model uncertainty**: generated descriptions and tags are useful starting
  points, not authoritative catalog records.
- **Archival footage quality**: digitized film can be black-and-white, damaged,
  low resolution, or visually unlike modern benchmark images.
- **Traceability**: reviewers need to know which config and model path produced a
  catalog.
- **Hiring-context legibility**: a technical reviewer should be able to inspect
  the system without private data or institutional access.

## What I built

The headline is a multimodal retrieval engine. Generated metadata is the
substrate; the retrieval stack is what makes the catalog searchable across text
and image:

- **SigLIP2-multilingual image/text embeddings** (`google/siglip2-large-patch16-256`,
  1024-d; default) with OpenCLIP ViT-B/32 retained as a legacy backend for
  comparison.
- **Hybrid retrieval** — dense SigLIP2 cosine similarity ⊕ BM25 over Moondream
  descriptions and curator tags, fused with Reciprocal Rank Fusion.
- **Cross-film visual rhymes** — kNN over keyframe embeddings, diversified with
  Maximal Marginal Relevance (MMR).
- **Cross-encoder reranker** — `BAAI/bge-reranker-v2-m3`, typed and wired into the
  search path but **default-OFF for v1.0** because its effect is unmeasured and its
  text-only design is suspect (see `RERANKER_DECISION.md`). It is not applied by default.

The retrieval engine sits on top of, and is supported by:

- A Python processing pipeline for FFprobe/FFmpeg inspection, frame extraction,
  PySceneDetect scene segmentation, visual analysis, embedding indexing, and local
  Moondream 2 scene descriptions.
- A FastAPI + HTMX web interface with Search, Scenes, Annotate, Processing, and
  About surfaces.
- Text search, reference-image search, scene browsing, and tag filtering over
  generated metadata.
- Manual annotation persistence with normalized curator tags that merge with
  generated metadata.
- A pluggable model backend registry using typed Protocols so model roles can be
  swapped without rewriting the pipeline.
- A reproducible public-demo scaffold using Library of Congress item `00694220`,
  *The Great Train Robbery* (1903).
- Retrieval evaluation tooling for Recall@5, Recall@10, MRR, nDCG@10, and
  manual annotation correction stats.
- Domain packs for `archive` and `media_broadcast` prompt/export behavior.
- Domain-aware JSON/CSV exports for downstream catalog review.
- Run manifests that record input identity, config hash, selected domain, model
  backend names, step outcomes, errors, and artifact paths.

## System design

The pipeline is intentionally modular:

1. **Ingest and inspect**: FFprobe records basic video properties and FFmpeg
   extracts frames.
2. **Segment**: PySceneDetect identifies scene boundaries and stores up to three
   representative keyframes per scene (configurable via `keyframes_per_scene`).
3. **Analyze**: visual-analysis backends detect faces, objects, and coarse
   environment labels.
4. **Describe**: the selected local vision-language backend writes scene
   descriptions and tags.
5. **Index**: SigLIP2 image embeddings are stored with scene/keyframe mapping so
   text and image search can run locally.
6. **Review**: the web UI presents generated metadata and lets humans correct
   scene-level tags.
7. **Measure and export**: evaluation reports quantify retrieval behavior;
   exports and manifests make the catalog auditable outside the app.

The architecture details are documented in [Architecture](ARCHITECTURE.md).
The operational behavior, generated artifacts, and known constraints are
documented in [Operations](OPERATIONS.md).

## Human review

The annotation tab is part of the product thesis. The AI output is valuable
because it creates a searchable first pass, but cultural-heritage metadata still
needs human judgment.

Manual tags are stored separately from generated tags, then merged at read time
for search and browsing. This preserves the distinction between model output and
human correction while still making curator work immediately useful.

Evaluation also treats manual corrections as a measurable signal. The M2
evaluation package can compare generated and manual tag sets and report accepted,
added, removed, and corrected tags. That gives the project a concrete answer to
"what did human review improve?"

## Evaluation

The evaluation path is designed to avoid vague AI claims. It uses a fixed query
file and a fixed demo index, then produces JSON and Markdown reports.

Implemented metrics:

- Recall@5
- Recall@10
- MRR
- nDCG@10
- manual annotation correction statistics

The current seed query set lives at
[`data/eval/archive_demo_queries.yaml`](../data/eval/archive_demo_queries.yaml).
The evaluation command is:

```bash
uv run python scripts/run_eval.py \
  --config config/demo.yaml \
  --queries data/eval/archive_demo_queries.yaml \
  --output-dir data/eval/reports
```

Final public numeric results should be published only after the final demo
artifact bundle, scene detection outputs, and query labels are verified together.
The evaluator and report format are already implemented; the final metric values
are a release step, not a code gap.

## Domain adaptation

The project is archive-first, but the implementation is not hardcoded as a
single archive app. Domain packs define:

- domain id and label,
- metadata fields,
- taxonomy values,
- prompt templates,
- filters,
- export mappings,
- sample output records.

The default `archive` pack preserves the film-cataloging behavior. The
`media_broadcast` pack shows how the same pipeline can support adjacent
workflows such as b-roll review, clip discovery, and licensing triage.

This is the main generalization strategy: keep the processing engine stable and
move domain language, prompts, and export shape into configuration.

See [Domain packs](DOMAIN_PACKS.md) for schema and examples.
A field-by-field comparison of the two packs is in [Domain packs](DOMAIN_PACKS.md#archive-vs-media_broadcast-side-by-side).

## Production signals

M4 added two important engineering surfaces:

- **Structured exports**: `/api/export/catalog.json` and
  `/api/export/catalog.csv` produce domain-aware scene catalogs that can be
  reviewed outside the app.
- **Run manifests**: each pipeline run records provenance about input identity,
  config hash, selected domain, model backend names, step states, errors, and
  artifact paths.

These do not turn the project into a hosted platform. They make the local ML
workflow easier to audit, reproduce, and discuss with technical reviewers.

The API surface is documented in [API reference](API.md). Release gates and
manual browser checks are documented in
[Release verification](RELEASE_VERIFICATION.md).

## Evidence map

| Claim | Evidence |
|---|---|
| Local-first visual-search workflow | [Privacy/offline notes](PRIVACY_OFFLINE.md), [Architecture](ARCHITECTURE.md) |
| Searchable scene catalog | `src/kuaa/pipeline.py`, `src/kuaa/embeddings.py`, FastAPI Search/Scenes routes |
| Human-in-the-loop correction | `api/services/annotations.py`, annotation tests, [Evaluation](EVALUATION.md) |
| Measured retrieval behavior | `src/kuaa/eval/`, `scripts/run_eval.py`, `tests/test_eval_metrics.py` |
| Domain adaptability | `config/domains/`, `src/kuaa/domain.py`, [Domain packs](DOMAIN_PACKS.md) |
| Exportable catalog | `src/kuaa/exporters/`, `api/routes/export.py`, `tests/test_exports.py` |
| Run provenance | `src/kuaa/run_manifest.py`, `tests/test_run_manifest.py` |
| Release discipline | [Release verification](RELEASE_VERIFICATION.md), test/lint/type gates |

## What changed after measurement

The evaluation work changed the project framing. Instead of claiming that
semantic search "works," the project now asks narrower questions:

- Which query intents return relevant scenes?
- Which expected scenes are consistently missed?
- Where do manual tags repair model output?
- Are generated tags useful enough for filtering, or mainly useful as a review
  starting point?
- Did a model or domain-prompt change improve search, or just change wording?

The final public evaluation report should include one strong result, one weak
result, and one concrete follow-up. That is more credible than a polished demo
with no failure analysis.

## Limitations

Current public-release limits are intentional and documented:

- The app is local-first and single-user.
- The app has a multi-film registry, but it is not a hosted multi-user service.
- Global prototype chrome has been removed from the launch UI; visible tool
  controls are expected to be backed by behavior.
- Health and readiness routes (`/health`, `/ready`) ship for ops monitoring;
  container packaging is out of scope — the project is `uv`-only with no Docker
  (see the program spec §16).
- Final demo screenshots, checksums, and metrics depend on the final published
  artifact bundle.
- Model licenses and downstream usage need review before institutional or
  commercial deployment.
- Generated descriptions can be wrong; the project treats them as reviewable
  first-pass metadata.

## Launch status

Ready now:

- public positioning docs,
- demo scaffold and validation script,
- evaluation runner and query seed,
- domain packs,
- exports,
- run manifests,
- API and operations docs,
- automated test coverage for the implemented surfaces.

Pending release verification:

- final published demo artifact ZIP,
- final artifact checksums,
- populated screenshots,
- final evaluation report from the published bundle,
- short demo video and technical walkthrough recording,
- GitHub release tag and release notes.

## Next steps

1. Publish and verify the demo artifact bundle.
2. Run `scripts/prepare_demo.py --check` against the extracted bundle.
3. Run the evaluation command and save the generated report.
4. Capture populated Search, Scenes, Annotate, Processing, and About screenshots.
5. Export JSON/CSV catalog files and preserve a sample `run_manifest.json`
   excerpt for release notes.
6. Record the short demo and technical walkthrough using the M5 scripts.
7. Tag the release after the automated and manual release gates pass.
