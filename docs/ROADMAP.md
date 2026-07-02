# Roadmap

This roadmap summarizes the current forward plan by capability rather than by
milestone number. For the fuller narrative (problem, architecture, evaluation,
domain packs, limitations, next steps), see [`CASE_STUDY.md`](CASE_STUDY.md).

## Implemented now

- Local video processing pipeline:
  - FFmpeg/FFprobe video inspection and frame extraction,
  - PySceneDetect scene detection and keyframe extraction,
  - visual analysis,
  - visual embeddings,
  - local Moondream 2 scene descriptions.
- Semantic search:
  - text-to-scene search with CLIP, BM25, and hybrid retrieval,
  - image-to-scene search,
  - tag-filtered search.
- Multi-film library:
  - registry-backed film list,
  - per-film artifact layout,
  - aggregate and per-film search/browse flows.
- Cross-film visual rhymes:
  - anchor deep links,
  - MMR diversity control,
  - echo-grid and inspector UI.
- Human annotation:
  - manual scene tags,
  - description editing,
  - annotation persistence,
  - tag merge with generated metadata.
- FastAPI + HTMX web interface:
  - Search,
  - Scenes,
  - Annotate,
  - Processing,
  - Rimas,
  - About.
- Processing job progress with server-sent events.
- Admin-gated eval grading UI with JSONL grade persistence and live metrics.
- Pluggable model backend registry using typed Protocols.
- Offline-oriented UI assets: local JavaScript, icons, fonts, and CSS.
- Domain-aware JSON/CSV exports and run manifests.
- Regression test coverage for web routes, services, search, processing, i18n,
  accessibility, and model protocol behavior.

Known UI wiring gaps are tracked internally. The launch policy is that visible
tool controls must be backed by current behavior; global prototype chrome has
been removed.

## Public baseline

Goal: make the project legible to recruiters, team leads, and open-source
visitors.

Planned work:

- English-first README pass.
- Screenshots and architecture diagram.
- Public project brief.
- Model inventory and license notes.
- Offline/privacy documentation.
- Roadmap and issue breakdown.

Status:

- Project brief: drafted.
- Architecture doc: drafted.
- Model inventory: drafted.
- Privacy/offline notes: drafted.
- Task breakdown: drafted.
- README discovery links: added.
- Full English-first README opening: completed first pass.

## Reproducible demo

Goal: let reviewers see a populated app quickly without private data.

Implemented:

- Primary demo source selected: Library of Congress item `00694220`, *The Great
  Train Robbery* (1903).
- Demo config added at `config/demo.yaml`.
- Demo manifest added at `data/demo/manifest.json`, with per-artifact
  checksums and a release URL already filled in (not placeholders).
- Demo preparation/validation script added at `scripts/prepare_demo.py`.
- Demo docs added: `docs/DEMO.md`, `docs/DEMO_DATA.md`, and demo verification
  notes.
- Populated UI screenshots captured under `docs/` (Scenes, Annotate,
  Processing, Search, Add-film, scene-detail).

Remaining release tasks:

- Confirm the artifact ZIP referenced in `data/demo/manifest.json` is actually
  reachable at its release URL (an operational/hosting step, not a code gap).
- Record the short demo walkthrough video.

Success criteria:

- A reviewer can open a populated UI within five minutes on a normal laptop.
- The demo does not require private museum data.
- The full end-to-end processing path remains documented separately.

## Evaluation harness

Goal: show that the AI system is measured, not just demonstrated.

Implemented:

- Query YAML schema defined in `docs/EVALUATION.md`.
- `data/eval/archive_demo_queries.yaml` with at least 20 public-demo queries.
- File-based evaluation package under `src/kuaa/eval/`.
- `scripts/run_eval.py` to load the configured demo index or a per-film
  library (`--film-slug`), run text/image/rhyme queries, and write JSON plus
  Markdown reports.
- Recall@5, Recall@10, MRR, and nDCG@10 metrics.
- Manual annotation correction stats tracked from `manual_annotations.json`.

Remaining release tasks:

- Run the evaluation against the final published demo artifact bundle.
- Review and refresh query labels if scene detection changes before release.
- Publish the generated `report.md` metrics in release notes or docs.

Success criteria:

- A fixed demo index and query file produce repeatable metrics.
- The report includes model/config identity.
- The project shows both strengths and failures.

## Domain packs

Goal: prove the system can adapt beyond film archives without becoming a set of
unrelated apps.

Implemented:

- Domain pack YAML schema defined (`config/domains/archive.yaml`,
  `config/domains/media_broadcast.yaml`).
- Archive-specific prompts, fields, filters, and export mapping moved behind
  the domain interface.
- `archive` domain pack as the default; `media_broadcast` as the first
  adjacent industry pack, including its own eval query set
  (`data/eval/media_broadcast_queries.yaml`).
- Current archive demo preserved as the default domain.

Remaining release tasks:

- Run an end-to-end description pass with a non-archive domain on sample media.
- Decide which domain-specific fields should become visible in the UI first.
- Expand public structured export endpoints for additional domains.

Success criteria:

- Switching domain config changes prompts and output schema without pipeline
  rewrites.
- The second domain has sample data, outputs, and evaluation queries.

## Production signals

Goal: demonstrate engineering maturity around the ML workflow.

Implemented:

- Domain-aware export package (`src/kuaa/exporters/`) builds a reloadable
  scene catalog from current metadata artifacts.
- JSON and CSV export routes are available in the local FastAPI app:
  `/api/export/catalog.json` and `/api/export/catalog.csv`.
- Run manifest writer (`src/kuaa/run_manifest.py`) records input identity,
  config hash, selected domain, model backend names, timestamps, step
  outcomes, errors, and output artifacts.
- Manifest writing is integrated into both pipeline execution paths:
  `CatalogPipeline.run()` and the Processing-tab selected-step worker.
- API documentation covers current REST/HTMX surfaces and export endpoints.
- Operations notes cover releases, failure behavior, export paths, manifest
  retention, and the current multi-film library layout.

Remaining release tasks:

- Verify exports against the final published demo artifact bundle.
- Include a sample `run_manifest.json` excerpt in release notes or the case
  study after the final demo run.
- Decide whether the launch package needs a tiny saved sample export under
  docs assets.

Success criteria:

- A generated catalog can be exported and reloaded.
- Model-generated results are traceable to run configuration and model revision.
- Failure behavior is documented and test-covered.

Deferred (not part of this phase):

- CPU Docker image after the `uv` release path and demo artifact bundle are
  stable.
- Collaboration/share/import/settings features behind the visible prototype
  chrome.
- Reranker dispatcher plumbing and per-modality eval scoring beyond the
  text-only report path.

## Launch package

Goal: turn the repo into public career-transition evidence.

Implemented:

- Case study (`docs/CASE_STUDY.md`) ties the problem, constraints,
  architecture, evaluation, domain packs, production signals, limitations, and
  next steps together.
- Launch-package verifier (`scripts/check_launch_package.py` +
  `tests/test_launch_package.py`) checks that required public docs exist and
  still avoid unfilled placeholders before release.

Still pending (confirmed missing by running the verifier today):

- Launch plan covering the public posts (origin, demo, evaluation, domain
  adaptation, architecture/production signals).
- Short general-audience demo video script and longer technical walkthrough
  outline.
- Resume bullets, LinkedIn featured-project copy, and recruiter-facing project
  summary.
- GitHub release notes draft covering demo artifacts, verification commands,
  known limits, and the final publish checklist.

Remaining release tasks:

- Write the launch plan, demo video script, resume bullets, and release notes
  draft listed above so the launch-package verifier passes.
- Publish the final demo artifact ZIP and confirm it is reachable at the
  manifest's release URL.
- Run final evaluation and copy metrics into release notes.
- Capture the two demo videos (short and technical walkthrough).
- Add final export and `run_manifest.json` excerpts to release notes.
- Complete live reranker dispatcher wiring before exposing a Buscar Rerank
  control.
- Tag the public release after automated and manual release gates pass.

Recommended first public launch:

- Launch after the reproducible demo bundle, evaluation report, screenshots, and
  release notes are verified against the same artifact bundle.
- Keep claims tied to implemented surfaces: local processing/search,
  multi-film library browsing, annotation, visual rhymes, evaluation, domain
  packs, exports, and run manifests.

## Demo release automation

Goal: make the final public demo artifact bundle reproducible instead of a
manual ZIP-and-checksum step.

Implemented:

- Deterministic demo bundle builder (`scripts/build_demo_bundle.py`) that
  validates `data/demo/runtime/` before packaging.
- Generates a release ZIP, bundle checksum file, per-artifact checksums, and
  an updated manifest preview, plus copy/paste release snippets for
  `data/demo/manifest.json` and GitHub release notes.
- Test coverage in `tests/test_demo_bundle_builder.py` (passing).
- Listed alongside the other operational scripts in `scripts/README.md`.

Remaining:

- A dedicated maintainer-facing write-up of the build/verify/upload/refresh
  workflow does not exist yet as a standalone doc (today `scripts/README.md`
  covers it at a summary level only).

Success criteria:

- The bundle builder fails before packaging when required demo artifacts are
  missing or invalid. — met.
- Rebuilding the same runtime tree produces a stable ZIP checksum. — met
  (fixed internal ZIP timestamp, deterministic ordering).
- The generated manifest preview includes the final bundle checksum and
  required runtime artifact checksums. — met.
- Release docs explain how this phase connects to the reproducible demo
  download/validation path, the evaluation harness, production-signal
  exports/manifests, and the launch package release notes. — not yet written
  as a standalone doc.

## Deferred work

- **Retrieval depth — cross-encoder reranker rework.** The reranker ships
  **disabled by default** for v1.0 ([`RERANKER_DECISION.md`](RERANKER_DECISION.md)):
  its value is currently unmeasured (the proxy ablation reranked empty
  descriptions via a core-path bug) and its text-only design is suspect on
  short captions. The forward fix is either (a) RRF-fuse the reranker as a
  vote instead of a hard replace, then re-ablate, or (b) a cross-modal
  VLM-as-judge that scores the keyframe image — or remove it outright. A
  portfolio-grade retrieval story needs one of these, not a measured-unknown
  component left default-off.
- **Audio modality (CLAP audio search + CLIP×CLAP cross-modal fusion)** —
  per-scene audio extraction + CLAP joint text+audio embeddings, audio-only
  search, and linear late-fusion of CLIP × CLAP scores under one tunable
  weight. Implemented and accepted on the demo film, then removed from `main`
  (2026-05-31) to focus the v1.0 surface on the visual/text retrieval story.
  The removal and rationale are recorded in project history.
- **Whisper dialogue transcription** — per-scene speech-to-text
  (faster-whisper) feeding a second BM25 lexical surface and a transcript view
  in the scene inspector. Prototyped in May 2026 and then removed from `main`
  to keep the v1.0 surface focused. Deferred to the v0.8-rc landmark.
