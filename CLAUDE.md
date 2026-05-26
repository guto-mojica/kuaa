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

The main branch is currently **migrating from v0.2.1 (Streamlit) to v0.3.0
(FastAPI + HTMX + Jinja2)**. The Streamlit app remains functional as
`app_streamlit.py` until feature parity is reached.

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
api/                 Thin HTTP layer. Each route calls src/cinemateca/ and returns JSON or HTML.
web/templates/       Jinja2. base.html + partials/ for HTMX fragments.
web/static/          CSS, vendored htmx.min.js, icons.
web/locales/         pt_BR and en, managed by Babel.
config/              default.yaml (versioned) + local.yaml (gitignored).
tests/               pytest, no heavy-model dependencies in test_smoke.
docs/                DESIGN_SYSTEM.md (visual source of truth), PROTOCOL_OPTION.md (pluggable model-backend design).
app.py               FastAPI entrypoint (uvicorn api.server:app).
app_streamlit.py     Legacy entrypoint. Removed at end of migration.
```

**Rule:** AI logic lives in `src/cinemateca/`. If you're about to write model inference,
video parsing, or embedding computation inside `api/` or `web/`, stop and move it to
`src/cinemateca/`.

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
- Update the migration tracker at the bottom of this file when relevant.

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
#   pip install -e ".[full]" && pip install pytest pytest-cov black ruff mypy

# Unified CLI (Typer) — discoverable via --help at every level.
uv run cinemateca --help                       # command tree
uv run cinemateca <cmd> --help                 # flags for any subcommand

# Run the FastAPI interface (v0.3.0+)
uv run cinemateca serve                        # http://127.0.0.1:8501 (--reload by default)
uv run cinemateca serve --port 9000 --no-reload
# Legacy `uv run app.py` still works (delegates to `cinemateca serve`).

# Run the legacy Streamlit interface (during migration)
uv run streamlit run app_streamlit.py

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
uv run black .
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

- `web/templates/**` during migration.
- `api/routes/**` during migration.
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

## v0.2.1 → v0.3.0 migration tracker

Last updated: in the commit message that touched CLAUDE.md.

- [x] Folder structure (`api/`, `web/`, `docs/`) created
- [x] `library.py` in `src/cinemateca/` to manage film collection
- [x] FastAPI app skeleton running on `localhost:8501`
- [x] Base layout (sidebar + main) in `web/templates/base.html`
- [x] Search tab functional via HTMX
- [x] Scenes tab functional via HTMX
- [x] Annotate tab functional via HTMX
- [x] Processing tab with SSE
- [x] About modal
- [x] i18n PT/EN extracted and translated
- [x] Streamlit parity confirmed → tagged `v0.2.1-streamlit-final`
- [x] uv adopted for env/deps (config-only, lockfile deferred — see docs/superpowers/specs/2026-05-16-uv-migration-scaffold-design.md)
- [x] FastAPI regression recovery (Phases 0–8): regressions fixed (full-page parity, Processing render, scene-id tag filtering, SSE close), service layer, pipeline gating/cancellation, single-film v0.3 (multi-film deferred), i18n/a11y/offline, tests 18→208 — see docs/RELEASE_VERIFICATION.md
- [x] Pluggable model backends (PROTOCOL_OPTION Steps 1–2): 5 Protocols +
  registry, all roles moved, hard cutover (no shims), describer replaced
  with keyless offline Moondream-2 GGUF (transformers removed) — see
  docs/superpowers/specs/2026-05-17-pluggable-model-backends-design.md
- [x] Describer default reverted to HF transformers Moondream2 @2025-01-09
  (deployment ease: keyless, GPU via prebuilt PyTorch wheels on Win/Mac/Linux,
  no source build). GGUF kept opt-in. transformers pinned >=4.44,<5 (tf5
  hard-fails for every moondream2 revision, verified). uv.lock now committed.
  See docs/superpowers/plans/2026-05-18-transformers-describer-default.md
- [x] Multi-film library: per-film dirs, native file picker, film registration
  and removal, `scan_library` multi-film aware, Processing dropdown synced to
  active-film cookie, slug read from `film.json`

Keep this list updated as steps complete.

---

## v0.3.0 → v1.0.0 launch tracker

4-month effort starting 2026-05-19. Full spec:
`docs/superpowers/specs/2026-05-19-multimodal-retrieval-and-launch-design.md`.
Dual purpose: take the project to a credible public v1.0 launch, and produce
a portfolio piece for the maintainer's applied-ML / retrieval career
transition. Scope is *not* locked — if risks fire, engineering can drop or
defer features; timeline extension is one option among several. Grilled
2026-05-24 (correction to an earlier "scope locked" framing).

### Month 1 — Foundation
- [x] **Multi-film library** — `films.json` registry; per-film `data/library/<slug>/`
  layout; cross-film search/browse + sidebar selector. Implemented across
  T1–T11 of `docs/superpowers/plans/2026-05-20-multi-film-library.md`
  (33 commits on `feat/multi-film-library`; suite 265 → 332 passing).
  Acceptance migration of real Jeca Tatu data still pending (T12, manual).
- [ ] Docker image, one-command run (CPU-default, GPU-optional)
- [ ] Hosted demo skeleton on HuggingFace Spaces (CPU tier)
- [x] CLAP integration kickoff — `AudioEmbedder` Protocol +
  `ClapHFEmbedder` (HF transformers, `laion/larger_clap_general`, 10s
  chunk + mean-pool, L2-normalised joint text+audio space) +
  `SceneAudioExtractor` (ffmpeg → 48 kHz mono PCM16 per scene) +
  two pipeline steps `audio_extract`/`audio_embed` (default OFF; opt-in
  via `--steps`). Real-data acceptance on Jeca Tatu: 412 scenes encoded
  in 111 s on CUDA (RTX 5090), `audio/clap_embeddings.npy` shape
  (412, 512) float32 L2-normalised, mapping schema verified. Suite
  332 → 413 passing on `feat/clap-audio-kickoff`. See
  `docs/superpowers/plans/2026-05-20-clap-audio-embeddings.md`.
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
  `docs/superpowers/specs/2026-05-24-deep-modules-refactor-design.md`;
  plan: `docs/superpowers/plans/2026-05-24-deep-modules-refactor-p1-search.md`.
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
  `docs/superpowers/specs/2026-05-24-deep-modules-refactor-design.md`;
  plan: `docs/superpowers/plans/2026-05-25-deep-modules-refactor-p2-library.md`.
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
  `docs/superpowers/specs/2026-05-24-deep-modules-refactor-design.md`;
  plan:
  `docs/superpowers/plans/2026-05-25-deep-modules-refactor-p3-services.md`.
- [ ] Pre-launch LinkedIn "I'm building this" post

### Month 2 — Retrieval depth + audio (HARD FREEZE on new features)
- [ ] CLAP audio embeddings complete; audio-only search in UI
- [ ] Whisper transcripts indexed (faster-whisper, `Transcriber` Protocol)
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
  `docs/superpowers/specs/2026-05-23-hybrid-search-design.md` +
  `docs/superpowers/plans/2026-05-23-hybrid-search.md`. F1 (eval ablation
  on Jeca Tatu) ainda pendente — corrida manual requer dados reais.
  Cross-encoder reranker + M-CLIP multilingual model shipped 2026-05-25 (see items below).
- [x] Cross-encoder reranker — `cinemateca.search.rerank`; `ms-marco-MiniLM-L-12-v2`
  (English default) / `mmarco-mMiniLMv2-L12-H384-v1` (multilingual, local.yaml).
  Top-50 candidates → top-k rerank; lazy-load + module-level cache; graceful fallback.
  Validated on Jeca Tatu: [old→new] rank logs confirm reshuffling. 2026-05-25.
- [x] Multilingual visual model — M-CLIP (`clip-ViT-B-32-multilingual-v1` via
  sentence-transformers). `MClipEmbedder` subclasses `OpenClipEmbedder`, overrides
  only `encode_text()`. Existing `keyframe_embeddings.npy` valid (same 512-dim space).
  `load_index` routes to registry; embedder name in cache key. Validated 2026-05-25.
- [ ] CLAP archival-audio sanity check (pre-commit gate on Jeca Tatu)

### Month 3 — Fusion + visual rhymes + eval annotation
- [ ] Cross-modal CLIP × CLAP fusion search
- [x] Visual rhymes (cross-film kNN + MMR diversity) — stub MVP shipped with
  Mojica redesign: cosine kNN cross-film over existing CLIP keyframe embeddings,
  `Rimas Visuais` tab fully wired (anchor + echoes UI). MMR / diversity
  controls and curated single-anchor refinements deferred to M3.
- [ ] 50–100 curator-annotated eval pairs
- [ ] Landing-page README draft; blog post outline

### Month 4 — Eval + writeup + launch
- [ ] Ablation table + per-modality breakdown
- [ ] Failure-mode analysis (5–10 queries explained)
- [ ] Technical blog post published (own domain + LinkedIn article)
- [ ] 90-second demo video published
- [ ] README + GitHub polish + `v1.0.0` release tag
- [ ] LinkedIn launch post

Keep this list updated as steps complete.

---

*This document is alive. When you notice it's outdated or imprecise, propose an edit.*
