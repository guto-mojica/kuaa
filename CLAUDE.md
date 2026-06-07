# CLAUDE.md

Operational briefing for Claude Code and any agent working on this repository.
Read this file before any action. For project context, see `README.md`.
For installation, see `SETUP.md`. For design and architecture decisions, see `docs/`.

This file is written in English to remain useful to collaborators across languages.
User-facing strings and product vocabulary are bilingual (PT/EN) — see the vocabulary
table below.

---

## 30-second summary

Cinemateca-imgsearch is an offline audiovisual cataloguing system for film archives.
It takes video files and produces searchable metadata (scenes, faces, objects,
natural-language descriptions, semantic embeddings).

The interface is **FastAPI + HTMX + Jinja2** (v0.3.0+). The legacy Streamlit
app was removed once parity was confirmed; the FastAPI surface is the only
supported UI.

---

## Canonical stack

| Layer | Technology | Location |
|---|---|---|
| AI core | Python 3.10+, PyTorch (CPU/MPS/CUDA), CLIP, Moondream 2, YOLOv8, MTCNN, PySceneDetect | `src/cinemateca/` |
| API | FastAPI + Pydantic | `api/` |
| Frontend | HTMX + Jinja2 + custom CSS (no build step) | `web/` |
| i18n | Babel + `.po` files | `web/locales/` |
| Config | YAML (default + local override) | `config/` |
| Tests | pytest | `tests/` |

**Decisions that must not be reversed without explicit discussion with the maintainer:**

- No React, Vue, Svelte, or any SPA framework with a build step. HTMX is the deliberate choice.
- No npm or node as a project dependency. Python + static HTML + vendored JS only.
- No cloud APIs for inference. All models run locally.

If you think one of these should change, open an Issue before touching anything.

`src/cinemateca/*` may be modified during the FastAPI regression-recovery work
(maintainer-approved 2026-05-16). The earlier "do not rewrite the core" rule is
lifted for this effort. The artefact-safety rule still applies independently:
changes that alter AI-model behaviour, prompts, or dependency versions can
invalidate generated artefacts — see "Constraints and care" and ask before
applying those.

---

## Project vocabulary

These words have fixed meaning in code, URLs, translation keys, and UI.
**Do not invent synonyms.**

| PT term | EN term | Meaning |
|---|---|---|
| Acervo | Library | Full collection of catalogued films |
| Filme | Film | A single audiovisual work in the library (parent entity) |
| Cena | Scene | Segment delimited by detected cuts |
| Keyframe | Keyframe | Representative frame of a scene |
| Buscar | Search | Semantic search tab (text or image) |
| Cenas | Scenes | Library browsing tab (was "Catálogo" in v0.2.x) |
| Anotar | Annotate | Manual tag curation tab |
| Processamento | Processing | Tab visible only when active jobs exist |
| Sobre | About | Institutional credits modal/page |
| Tag | Tag | Label applicable to a scene (automatic or manual) |
| Pipeline | Pipeline | Step sequence: frames → scenes → visual → embeddings → llm |
| Rimas | Rhymes | Cross-film visual similarity matches (Rimas Visuais tab) |
| Âncora | Anchor | The scene whose visual rhymes are being explored |

Terms to avoid because they carry domain ambiguity:

- **Catalogue/Catálogo** — used in v0.2.x for the browse tab; deprecated in v0.3.0
  because the catalogue is the whole system, not one tab.
- **Analysis/Análise** — ambiguous between "visual analysis" (module) and "full analysis"
  (pipeline). Use "visual analysis" or "pipeline" as appropriate.
- **Ingest/Ingerir** — OAIS technical term, awkward in Portuguese. Replaced by
  "add film" as a gesture and "Processing" as the status surface.
- **Pesquisar** — replaced by "Buscar" in Portuguese strings.

---

## Repository layout

```
src/cinemateca/      AI core. HTTP-agnostic logic. Cleanly importable.
  search/              4-verb / 7-type retrieval API (find / aggregate / reindex_bm25 / rerank).
  library/             Registry + scan + FilmContext + per-film metadata loaders.
  annotations/         Manual tags + descriptions + annotate-tab scene builders.
  rhymes/              Cross-film visual-rhyme algorithm + enrichment.
  eval/                Eval-set datasets + grades + IAA / κ metrics.
  retrieval/           BM25Index + RRF fusion primitives.
  models/              Protocol-typed model backends + registry.
api/                 Thin HTTP layer. Each route calls src/cinemateca/ and returns JSON or HTML.
web/templates/       Jinja2. base.html + partials/ for HTMX fragments.
web/static/          CSS, vendored htmx.min.js, icons.
web/locales/         pt_BR and en, managed by Babel.
config/              default.yaml (versioned) + local.yaml (gitignored).
tests/               pytest, no heavy-model dependencies in test_smoke.
docs/                DESIGN_SYSTEM.md (visual source of truth), PROTOCOL_OPTION.md (pluggable model-backend design).
app.py               FastAPI entrypoint (uvicorn api.server:app).
```

**Rule:** AI logic lives in `src/cinemateca/`. If you're about to write model inference,
video parsing, or embedding computation inside `api/` or `web/`, stop and move it to
`src/cinemateca/`.

**Deep-modules invariant (P1/P2/P3 refactor):** the `cinemateca.*` packages
must not import from `api/*`. This is enforced by `import-linter` (see
`.importlinter`); the matching LOC budget per service module is enforced
by `scripts/check_loc_budget.py`. Both run in CI
(`.github/workflows/refactor-guards.yml`). Public surface of each package
lives in its `__init__.py`; everything else is implementation detail.

---

## Git workflow

Claude Code reads the **local working tree** only. It does not automatically pull
from or push to GitHub. The discipline below avoids stale-code conflicts.

**Session opening — every time:**

1. `git status` — understand current state, see if there are uncommitted changes.
2. `git pull origin main` (or the working branch) — sync local with remote.
   Skip this and you may work against stale code, producing conflicts later.
3. `git log -5 --oneline` — know what was committed recently.
4. Read the file(s) you plan to modify before proposing changes.

**During the session:**

- For changes in `src/cinemateca/`, run `pytest tests/test_smoke.py` before
  and after to ensure nothing broke.
- For changes that touch the AI pipeline (model download, dependency version bump,
  Moondream prompt alteration), **ask the user before applying** — these changes
  may invalidate already-generated artefacts.

**Session closing:**

- Commit logically grouped changes with clear messages.
  Format: `type(scope): short description`
  Types: `feat`, `fix`, `refactor`, `docs`, `chore`, `test`.
  Example: `feat(api): add /api/library/tree endpoint with film grouping`.
- Push when the user confirms (`git push origin main` or to a working branch).
- If working on a feature branch, mention it in the commit summary so the user
  knows what to merge later.
- Update the launch tracker at the bottom of this file when relevant.

**What Claude Code will not do without explicit user request:**

- Push to remote.
- Force-push or rewrite history (`git rebase -i`, `git push --force`).
- Delete branches.
- Modify git tags.
- Resolve merge conflicts unilaterally — surface them to the user.

---

## Common commands

```bash
# One-time setup (uv — primary)
uv venv                       # creates .venv using the .python-version pin
uv sync --extra full --group dev   # full extra + dev tooling group
# Fallback without uv (no lockfile, so this still works identically):
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install -e ".[full]" && pip install pytest pytest-cov ruff mypy

# Unified CLI (Typer) — discoverable via --help at every level.
uv run cinemateca --help                       # command tree
uv run cinemateca <cmd> --help                 # flags for any subcommand

# Run the FastAPI interface (v0.3.0+)
uv run cinemateca serve                        # http://127.0.0.1:8501 (--reload by default)
uv run cinemateca serve --port 9000 --no-reload
# `uv run app.py` still works (delegates to `cinemateca serve`).

# Tests
uv run pytest tests/ -q
uv run pytest tests/test_smoke.py -v           # smoke only, no heavy models

# Single-film pipeline
uv run cinemateca info data/raw/jeca_tatu.mp4
uv run cinemateca process data/raw/jeca_tatu.mp4
uv run cinemateca process data/raw/jeca_tatu.mp4 --steps scenes,embeddings
uv run cinemateca process data/raw/jeca_tatu.mp4 --slug jeca_tatu   # match an existing slug

# Library-wide operations (use the REGISTERED slug — no filename→slug drift)
uv run cinemateca library list                            # registered films + state
uv run cinemateca library reembed                          # all films, --steps embeddings
uv run cinemateca library reembed --only jeca_tatu --steps embeddings,visual
uv run cinemateca library reembed --keep-existing          # partial reruns
uv run cinemateca library delete <slug> [--yes]            # destructive

# Config introspection
uv run cinemateca config show                              # merged effective YAML

# i18n
uv run pybabel extract -F web/babel.cfg -o web/locales/messages.pot web/
uv run pybabel update -i web/locales/messages.pot -d web/locales/
uv run pybabel compile -d web/locales/
# NOTE: `pybabel compile` rewrites every .mo with a fresh POT-Creation-Date
# header even when no translation changed, so it always dirties the tree.
# Only recompile when a .po actually changed. A .mo diff that is timestamp-
# header-only (identical msgstrs) is noise — discard it, don't commit it.

# Lint / format / typecheck (run before committing)
uv run ruff format .
uv run ruff check .
uv run mypy src
```

---

## Constraints and care

**Expensive operations — require explicit user confirmation before running:**

- `cinemateca process <video>` on any new video (~60–120min on CPU).
- Embedding regeneration (`cinemateca library reembed`) for the full library.
- LLM description regeneration (`--steps llm`) — slowest pipeline step.
- First-time model downloads (~2GB for Moondream).

For quick development checks, use the test film **Jeca Tatu (1959)** already
processed in `data/`. Artefacts persist across runs.

**Note:** the default describer is now HF-transformers Moondream2 (GPU via a
prebuilt PyTorch CUDA/MPS wheel — no source build). The GGUF describer below
is the opt-in alternative (`scene_describer: moondream_gguf`).

**GPU acceleration for the GGUF describer** — the slow CPU runs above become
~10–25× faster on the local NVIDIA GPU, but `llama-cpp-python` must be built
from source with CUDA, which requires a non-obvious toolchain workaround on
this machine (a one-line CUDA-header patch + `gcc15`). Full rationale, apply
/ verify / revert steps, and what to redo after a driver/kernel/CUDA/glibc
update are documented in **`docs/GPU_LLAMA_CPP_CUDA_BUILD.md`**. Read that
before touching the CUDA toolchain or rebuilding `llama-cpp-python`. The
describer's `config/default.yaml` → `llm.gpu_layers: -1` knob is a no-op on a
CPU-only build, so it is safe to leave enabled either way.

**Never modify without explicit instruction:**

- `config/local.yaml` — machine-specific, contains real paths.
- Anything under `data/` — user's archive.
- `models/` — downloaded models.
- Git version tags (`v0.x.x`).

**Files expected to change frequently:**

- `web/templates/**` for UI / partial work.
- `api/routes/**` for endpoint work.
- `CHANGELOG.md` with each significant feature.

---

## Coding conventions

- **Type hints required** on public signatures in `src/cinemateca/` and `api/`.
- **Docstrings in English** for public functions (project-wide collaboration).
  Inline comments may be in either language.
- **Black + Ruff + mypy** (configured in `pyproject.toml`). Ruff covers isort
  (`I`) rules — there is no separate isort. Run all three before committing.
- **Logging via `logging.getLogger(__name__)`**, never `print()` in production code.
- **Paths via `pathlib.Path`**, never string concatenation.
- **Config via dependency injection** in `api/` (use `api/deps.py`), never direct imports.

For HTML/Jinja:

- Full-page templates in `web/templates/*.html`.
- HTMX fragments in `web/templates/partials/*.html` (returned by HTMX endpoints).
- Element IDs use `kebab-case`; CSS classes use `kebab-case`.
- All user-visible strings must pass through `{{ _("...") }}` for i18n.

---

## Design ↔ Code workflow

The project operates in a hybrid agentic mode:

1. **Exploratory design** happens in chat conversations with Claude (artifacts) —
   new layouts, interaction patterns, palette. Outputs: mockup + description.
2. **Execution design** happens here in Claude Code — small variations, adjustments,
   polishing of existing components.
3. **Visual source of truth** is `docs/DESIGN_SYSTEM.md`. Colors, typography,
   spacing, canonical components.

When a design decision lands in code, update `docs/DESIGN_SYSTEM.md` if it establishes
a new pattern. Don't duplicate here — reference.

---

## When you (the agent) don't know

- About **product/UX decisions**: ask the user, don't infer.
- About **coding conventions not covered here**: follow what already exists in the repo.
- About **AI model changes**: stop and ask. Changing Moondream's prompt or CLIP's
  version invalidates artefacts across the whole library.
- About **new dependencies**: justify in the PR. Fewer dependencies preferred.

---

## v0.2.1 → v0.3.0 migration

**Complete.** The Streamlit UI was retired once FastAPI + HTMX reached
parity; `app_streamlit.py` and the `streamlit` / `gui` dependency entries
are gone. Tag `v0.2.1-streamlit-final` is the last Streamlit-era commit
for anyone who needs to recover historical UI behaviour. Major milestones
along the way (folder restructure, pluggable model backends, transformers
Moondream2 default, multi-film library) are captured commit-by-commit in
`CHANGELOG.md` and `docs/RELEASE_VERIFICATION.md`. Forward work continues
under the v0.3.0 → v1.0.0 launch tracker below.

---

## v0.3.0 → v1.0.0 launch tracker

4-month effort starting 2026-05-19. Full spec:
(internal design notes).
Dual purpose: take the project to a credible public v1.0 launch, and produce
a portfolio piece for the maintainer's applied-ML / retrieval career
transition. Scope is *not* locked — if risks fire, engineering can drop or
defer features; timeline extension is one option among several. Grilled
2026-05-24 (correction to an earlier "scope locked" framing).

> **Forward plan:** the single source of truth for remaining work is the
> presentation-refactor program spec
> (`docs/superpowers/specs/2026-05-29-presentation-refactor-design.md`, §13 maps
> these tracker checkboxes to workstream IDs WS-1…WS-6). Docker is **cut** (uv-only,
> §16); HuggingFace Spaces is **re-scoped** to an optional buildpack-PaaS stretch;
> Whisper transcription is **cut/deferred to v0.8-rc** (§16); the **CLAP audio +
> CLIP×CLAP fusion** modality is **cut** (2026-05-31, §16; removal recorded in
> `docs/archive/2026-05-31-audio-feature-extraction.md`). See `docs/README.md`
> for the documentation map.
>
> **Status (2026-05-31): the presentation-refactor program is COMPLETE.** All six
> workstreams (WS-1…WS-6) merged to `main`, every CI gate green; the drop-tier
> (C7/C12/U9/U10) and WS-2 cosmetic (A7/A10) items are CONFIRMED CUT (spec §16/§17). The
> remaining v1.0 work is **human-gated launch only**: curator eval grades (E5), the 90s demo
> video + screenshots (D8), the blog post, the `v1.0.0` release tag, and the LinkedIn posts.

### Month 1 — Foundation
- [x] **Multi-film library** — `films.json` registry; per-film `data/library/<slug>/`
  layout; cross-film search/browse + sidebar selector. Implemented across
  T1–T11 of (internal design notes)
  (33 commits on `feat/multi-film-library`; suite 265 → 332 passing).
  Acceptance migration of real Jeca Tatu data still pending (T12, manual).
- [x] ~~Docker image, one-command run~~ → **cut (§16); replaced by `uv` reproducible-run hardening (WS-5 T8)**: `uv sync --extra full --group dev` + `uv run cinemateca serve`, verified by `scripts/verify_fresh_run.sh`.
- [ ] Hosted demo — **re-scoped (§16)**: HF Spaces needs the Docker SDK (conflicts with no-Docker). Demo leads with the recorded 90s video + local `uv` run; a hosted instance is an optional stretch on a buildpack PaaS (Render/Railway, no in-repo Dockerfile). Not a committed v1.0 deliverable.
- [x] ~~CLAP integration kickoff — `AudioEmbedder` Protocol + `ClapHFEmbedder` +
  `SceneAudioExtractor` + `audio_extract`/`audio_embed` steps~~ → **CUT
  (2026-05-31, §16).** Implemented and accepted on Jeca Tatu (412 scenes in
  111 s on CUDA), then removed from `main` along with the whole audio modality
  to focus the v1.0 launch surface; preserved on `backup/pre-audio-removal`.
  Removal recorded in `docs/archive/2026-05-31-audio-feature-extraction.md`.
- [x] Eval annotation tool (FastAPI page behind admin flag; 5-sample validation)
  — lands as `/eval` at admin-flagged route; standalone 3-pane UI + JSONL grade
  persistence + P@K / nDCG / inversions / Cohen's κ metrics + keyboard router
  (0/1/2/3/S, j/k, ⌘⏎) + `cinemateca eval seed` CLI complete on
  `worktree-mojica-redesign`. M3 curator-pair seed work continues.
- [x] **Deep-modules refactor P1 — `cinemateca.search`** — extracted from
  `api/services/search.py` (1388 → **235** LOC, cap 250) and
  `api/routes/search.py` (471 → **148** LOC, cap 150) into a typed
  4-verb / 7-type public API (`find`, `aggregate`, `reindex_bm25`,
  `rerank` + `Query` / `Filters` / `HybridWeights` / `Hit` /
  `SearchResult` / `SearchMode` / `UploadRejected`). 14 files under
  `src/cinemateca/search/`. Layer rules enforced by CI (`import-linter`
  contract `cinemateca → api forbidden` + `scripts/check_loc_budget.py`).
  Behavior preserved byte-for-byte via 8 hermetic snapshot tests. Spec:
  (internal design notes).
- [x] **Deep-modules refactor P2 — `cinemateca.library`** — extracted
  from `src/cinemateca/library.py` (217 LOC) + `api/services/film_context.py`
  (138 LOC, **deleted**) + the data-access half of `api/services/catalog.py`
  into a 6-file package (~620 LOC): `registry.py` + `scan.py` + `context.py`
  + `paths.py` + `metadata.py` + `__init__.py` (with the new typed `Library`
  handle: `list/get/register/remove/context/state` methods). `api/services/
  catalog.py`: 403 → **250 LOC** (exactly at cap; no longer exempted). Six
  `cinemateca → api.services.*` carve-outs deleted from `.importlinter`;
  2 remain as documented P5 follow-ups (`aggregate -> services.search`,
  `_dispatch -> api.deps`). Public surface: 16+ names in
  `cinemateca.library.__all__` (`Library`, `Film`, `FilmContext`,
  `list_films`, `get_film`, `register_film`, `remove_film`,
  `scan_library`, `library_state`, `load_registry`, `save_registry`,
  `load_json`, `keyframe_url`, `to_smpte`, `derive_fps`,
  `load_tag_index`, `load_metadata`). Behavior preserved — verified by
  the 8 P1 snapshots + 17 new tests (7 in `test_library_scan.py` + 10
  in `test_library_handle.py`); full suite **774 passing**. Spec:
  (internal design notes).
- [x] **Deep-modules refactor P3 — services extraction** — three
  subsystems extracted (`cinemateca.annotations`, `cinemateca.rhymes`,
  `cinemateca.eval`). Three services slimmed: `annotations.py` 577 →
  **129**, `rhymes_service.py` 470 → **194**, `eval_service.py` 564 →
  **244** (all removed from LOC budget EXEMPTIONS). `FilmContext.from_paths`
  constructor added + `Library.context` raises `KeyError` (aligned with
  `Library.get_film`; eliminates the SimpleNamespace workaround flagged
  in P2 review). `.importlinter` zero carve-outs: both P5 follow-ups
  from P1/P2 reviews (`aggregate -> services.search`,
  `_dispatch -> api.deps`) resolved by moving helpers to
  `cinemateca.search.aggregate` and parametrising BM25 tunables as
  kwargs in `_dispatch.find()`. Suite: **774 → 777 passing**. Spec:
  (internal design notes).
- [ ] Pre-launch LinkedIn "I'm building this" post

### Month 2 — Retrieval depth + audio (HARD FREEZE on new features)
- [x] ~~CLAP audio embeddings complete; audio-only search in UI (`/api/search?modality=audio`)~~ → **CUT (2026-05-31, §16).** Shipped end-to-end (CLAP joint text+audio space, Áudio chip, PT/EN i18n, Jeca Tatu regression snapshot) then removed from `main` with the whole audio modality. Preserved on `backup/pre-audio-removal`; removal recorded in `docs/archive/2026-05-31-audio-feature-extraction.md`.
- [x] ~~Whisper transcripts indexed (faster-whisper, `Transcriber` Protocol)~~ → **cut from v1.0; deferred to v0.8-rc (§16).** Prototyped (`dc2c8f8`) then removed from `main` (2026-05-30) to keep the launch surface focused. The CLAP audio modality that briefly covered audio for launch was itself cut on 2026-05-31 (see above).
- [x] Hybrid search (CLIP ⊕ BM25, Reciprocal Rank Fusion) — shipped 2026-05-23
  on `worktree-hybrid-search-spec`. New package `src/cinemateca/retrieval/`
  (tokenize + corpus + BM25Index + RRF) feeds `search_hybrid()` orchestrator;
  per-FilmContext loader with 3-file mtime+size cache. `/api/search` defaults
  to `retriever=hybrid`; `?retriever=clip` is the regression pin (snapshot in
  `tests/fixtures/hybrid_search_regression.json`). Aggregate cross-film honors
  the same retriever mode. UI: Alpine popovers for Híbrido + k knobs (bring-
  forward of `alpine.min.js` from the Mojica branch); Rerank/MMR are obvious
  read-only chips with M2/M3 micro-badges. Suite +30 tests (583 → ~625
  passing on this branch). Spec/plano em
  (internal design notes). F1 (eval ablation
  on Jeca Tatu) ainda pendente — corrida manual requer dados reais.
  **Próximo:** M2 #4 cross-encoder reranker.
- [x] Cross-encoder reranker (text default; VLM-as-judge opt-in) — lands in
  `cinemateca.search.rerank` (BAAI/bge-reranker-v2-m3 default, `model="noop"`
  escape hatch); `apply_reranker(result, *, cfg)` service wrapper ready
  (UI affordance live; live wiring in production dispatchers tracked as
  follow-up Task 3.2b — default off). **v1.0 decision (2026-05-31): shipped
  DISABLED by default (`retrieval.reranker.enabled: false`) as tracked debt —
  its effect is unmeasured (the proxy ablation reranked empty descriptions) and
  the design is suspect. Must be fixed properly (RRF-fuse / VLM-as-judge) or
  removed before the portfolio is "finished." See `docs/RERANKER_DECISION.md`.**
- [x] Multilingual visual model (SigLIP-multilingual; M-CLIP fallback) —
  `cinemateca.models.clip.siglip_multilingual.SiglipMultilingualEmbedder`
  registered behind `models.image_embedder`; library uniformly re-embedded
  with `google/siglip2-large-patch16-256` (1024 dim); CLIP backups preserved
  as `.clip_openclip.npy` per-film for rollback (SigLIP2-large-256 substituted
  for the plan's invented siglip-large-multilingual id; library uniformly
  migrated). Registry-dispatch fix in `cinemateca.search.{cache,aggregate}`
  so query-time text encoder honours the config flag.
- [x] ~~CLAP archival-audio sanity check (`cinemateca eval clap-sanity`)~~ →
  **CUT (2026-05-31, §16)** with the audio modality. Removal recorded in
  `docs/archive/2026-05-31-audio-feature-extraction.md`.

### Month 3 — Fusion + visual rhymes + eval annotation
- [x] M3 pre-flight (close M2 leftovers) — Phases 0–5 shipped on
  `m3-preflight`: ~~audio search end-to-end~~ (CUT 2026-05-31), reranker stub
  filled in (bge-reranker-v2-m3, service wrapper, UI affordance — production
  dispatcher wiring tracked as follow-up Task 3.2b), SigLIP2-multilingual
  rolled out library-wide, ~~CLAP archival sanity gate~~ (CUT 2026-05-31). See
  (internal design notes).
- [x] ~~Cross-modal CLIP × CLAP fusion search (`?modality=fusion&w=0.5`)~~ →
      **CUT (2026-05-31, §16).** Shipped (linear late-fusion `score = w·clip +
      (1-w)·clap`, `visual ↔ audio` weight popover, Jeca Tatu regression pin)
      then removed from `main` with the whole audio modality. Preserved on
      `backup/pre-audio-removal`; removal recorded in
      `docs/archive/2026-05-31-audio-feature-extraction.md`.
- [x] Visual rhymes (cross-film kNN + MMR diversity) — MMR live in M3:
  `?lambda=` + `?k_candidates=` query params on `/tab/rimas`,
  `/api/rimas/echoes`, `/api/rimas/inspector`; UI Diversidade popover
  with range slider fires live HTMX updates on the echo grid. Default
  λ=0.5 / k_candidates=30 via `cfg.retrieval.rhymes`. Acceptance check
  on real 2-film library (10 tests) confirms cross_film_only +
  MMR-runs-cleanly + unique-scene diversification. Curated single-anchor
  refinements still deferred to M4 stretch.
- [x] M3 eval data infrastructure (M3 #3) — curated bilingual queries in
      `data/eval/m3_full_queries.yaml` (originally 15 text · 10 image · 10 audio
      · 10 fusion · 5 rhyme per spec §7.1; the 20 audio + fusion entries were
      retired with the audio cut on 2026-05-31, leaving **30 scorable** —
      15 text · 10 image · 5 rhyme); text-only subset `m3_text_queries.yaml`
      runs end-to-end through `scripts/run_eval.py` against Jeca Tatu CLIP index
      (R@5=0.189, MRR=0.254 on pre-curator hypotheses); `cinemateca eval seed`
      writes candidate slates; protocol in `docs/EVAL_PROTOCOL.md`;
      `scripts/freeze_eval_run.sh` snapshots grades as SHA256 tarballs;
      `.gitignore` covers per-run output (`*-run-*/`, `*-run-*.queries.json`,
      `*-run-*.jsonl`, `*.frozen-*.tar.gz`). **Curator annotation sessions
      are the remaining human-gated work** — surface in M4 standups until
      complete. Non-text query types (image / rhyme) have per-modality slate
      generators so `run_eval.py` scores them.
- [ ] 50–100 curator-annotated eval pairs — grading sessions on the M3 slate
- [x] Landing-page README draft; blog post outline; demo-video scope —
      the README draft was folded into `README.md` (D6; archived at
      `docs/archive/README_DRAFT.md`); see also
      `docs/launch/BLOG_OUTLINE.md`, `docs/DEMO_VIDEO_SCRIPT.md`.
      Drafts cite real shipped backends; metric numbers (P@K, MRR,
      nDCG) stay TBD until the M4 ablation table lands. M4 finalises
      copy + records the 90s video.

### Month 4 — Eval + writeup + launch
- [x] Ablation table + per-modality breakdown — WS-4 E2 (proxy-first ablation,
  KI/PR/HY labels; retriever-variant rows on the SigLIP2 default —
  CLIP/BM25/hybrid/hybrid+rerank/OpenCLIP; the `fusion` row was dropped with the
  audio cut on 2026-05-31) in `docs/EVALUATION_RESULTS.md`
  §M4; per-modality scoring (E3) via `run_eval --modality {text,image,rhyme,all}`.
  Numbers are HY-proxy on the Jeca Tatu corpus; they upgrade to human-validated
  on curator grades (`eval export` → `run_ablation --grades`). Evidence: hybrid
  (R@5 0.467) > CLIP (0.444); SigLIP2 > OpenCLIP; rerank *hurts* on short
  captions → confirms C5 default-OFF.
- [x] Failure-mode analysis (5–10 queries explained) — WS-4 E4: 8 worst-query
  cases with real Moondream captions + cross-retriever ranks in
  `docs/FAILURE_ANALYSIS.md` §M4. Key finding: HY label-coverage (not retrieval)
  is the dominant nDCG ceiling → curator grading is the real fix.
- [ ] Technical blog post published (own domain + LinkedIn article)
- [ ] 90-second demo video published
- [ ] README + GitHub polish + `v1.0.0` release tag
- [ ] LinkedIn launch post

Keep this list updated as steps complete.

### Tracker → workstream map (per spec §13)

| Tracker item | Maps to |
|---|---|
| Docker image / one-command run | **cut**; `uv` fresh-run = WS-5 **T8** |
| Hosted demo (HF Spaces) | **re-scoped** §16; optional buildpack stretch |
| Whisper transcripts indexed | **cut** → deferred to v0.8-rc (§16) |
| CLAP audio search / CLIP×CLAP fusion | **cut** (2026-05-31, §16); `backup/pre-audio-removal` |
| Cross-encoder reranker | WS-1 **C5** (typed; **default-OFF for v1.0 by decision — `docs/RERANKER_DECISION.md`**; WS-4 rerank ablation was confounded by an empty-description core-path bug, so its effect is unmeasured) |
| Ablation table / failure-mode analysis | WS-4 **E2 / E4 / E8** |
| Curator-annotated eval pairs | WS-4 **E5** `[HUMAN]` (proxy fallback E2 so launch isn't hard-blocked) |
| Blog / 90s video / `v1.0.0` tag / LinkedIn | WS-6 **D6 / D8**, gate **G5** |

---

*This document is alive. When you notice it's outdated or imprecise, propose an edit.*
