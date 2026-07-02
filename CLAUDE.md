# CLAUDE.md

Operational briefing for Claude Code and any agent working on this repository.
Read this file before any action. For project context, see `README.md`.
For installation, see `SETUP.md`. For design and architecture decisions, see `docs/`.

---

## 30-second summary

KUAA is an offline audiovisual cataloguing system for film archives.
It takes video files and produces searchable metadata (scenes, faces, objects,
natural-language descriptions, semantic embeddings).

The interface is **FastAPI + HTMX + Jinja2**. The AI core is HTTP-agnostic and
lives entirely in `src/kuaa/`.

---

## Canonical stack

| Layer | Technology | Location |
|---|---|---|
| AI core | Python 3.10+, PyTorch (CPU/MPS/CUDA), SigLIP2, Moondream 2, YOLOv8, MTCNN, PySceneDetect | `src/kuaa/` |
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
| Cenas | Scenes | Library browsing tab |
| Anotar | Annotate | Manual tag curation tab |
| Processamento | Processing | Tab visible only when active jobs exist |
| Sobre | About | Institutional credits modal/page |
| Tag | Tag | Label applicable to a scene (automatic or manual) |
| Pipeline | Pipeline | Step sequence: frames → scenes → visual → embeddings → llm |
| Rimas | Rhymes | Cross-film visual similarity matches (Rimas Visuais tab) |
| Âncora | Anchor | The scene whose visual rhymes are being explored |

Terms to avoid:

- **Catalogue/Catálogo** — the catalogue is the whole system, not one tab.
- **Analysis/Análise** — ambiguous. Use "visual analysis" or "pipeline" as appropriate.
- **Ingest/Ingerir** — replaced by "add film" as a gesture and "Processing" as the status surface.
- **Pesquisar** — replaced by "Buscar" in Portuguese strings.

---

## Repository layout

```
src/kuaa/      AI core. HTTP-agnostic logic. Cleanly importable.
  search/              4-verb / 7-type retrieval API (find / aggregate / reindex_bm25 / rerank).
  library/             Registry + scan + FilmContext + per-film metadata loaders.
  annotations/         Manual tags + descriptions + annotate-tab scene builders.
  rhymes/              Cross-film visual-rhyme algorithm + enrichment.
  eval/                Eval-set datasets + grades + IAA / κ metrics.
  retrieval/           BM25Index + RRF fusion primitives.
  models/              Protocol-typed model backends + registry.
api/                 Thin HTTP layer. Each route calls src/kuaa/ and returns JSON or HTML.
web/templates/       Jinja2. base.html + partials/ for HTMX fragments.
web/static/          CSS, vendored htmx.min.js, icons.
web/locales/         pt_BR and en, managed by Babel.
config/              default.yaml (versioned) + local.yaml (gitignored).
tests/               pytest, no heavy-model dependencies in test_smoke.
docs/                Design, architecture, eval, and ops docs (markdown only).
site/                Public GitHub Pages site (HTML/CSS/images), deployed via
                     .github/workflows/pages.yml. Not app code — keep app UI in web/.
app.py               FastAPI entrypoint (uvicorn api.server:app).
```

**Rule:** AI logic lives in `src/kuaa/`. If you're about to write model inference,
video parsing, or embedding computation inside `api/` or `web/`, stop and move it to
`src/kuaa/`.

**Deep-modules invariant:** the `kuaa.*` packages must not import from `api/*`.
Enforced by `import-linter` (`.importlinter`) and per-module LOC budgets
(`scripts/check_loc_budget.py`), both run in CI.

---

## Git workflow

**Session opening:**

1. `git status` — understand current state.
2. `git pull origin main` — sync with remote before editing.
3. `git log -5 --oneline` — know recent context.
4. Read the file(s) you plan to modify before proposing changes.

**During the session:**

- For changes in `src/kuaa/`, run `pytest tests/test_smoke.py` before and after.
- For changes that touch the AI pipeline (model download, dependency version bump,
  Moondream prompt alteration), **ask the maintainer before applying** — these
  changes may invalidate already-generated artefacts.

**Commit format:** `type(scope): short description`
Types: `feat`, `fix`, `refactor`, `docs`, `chore`, `test`.

**Never add `Co-Authored-By` trailers to commits.** The maintainer is the sole author.

---

## Common commands

```bash
# One-time setup
uv venv
uv sync --extra full --group dev

# Run the app
uv run kuaa serve                        # http://127.0.0.1:8501
uv run kuaa serve --port 9000 --no-reload

# Tests
uv run pytest tests/ -q
uv run pytest tests/test_smoke.py -v

# Single-film pipeline
uv run kuaa process data/raw/myvideo.mp4
uv run kuaa process data/raw/myvideo.mp4 --steps scenes,embeddings

# Library operations
uv run kuaa library list
uv run kuaa library reembed --only <slug> --steps embeddings
uv run kuaa library delete <slug> [--yes]

# Config
uv run kuaa config show

# i18n
uv run pybabel extract -F web/babel.cfg -o web/locales/messages.pot web/
uv run pybabel update -i web/locales/messages.pot -d web/locales/
uv run pybabel compile -d web/locales/
# NOTE: pybabel compile always rewrites .mo headers. Only recompile when a .po changed;
# a diff that is timestamp-header-only is noise — don't commit it.

# Lint / format / typecheck
uv run ruff format .
uv run ruff check .
uv run mypy src
```

---

## Constraints and care

**Expensive operations — confirm with the user before running:**

- `kuaa process <video>` on any new video (~60–120 min on CPU).
- `kuaa library reembed` for the full library.
- `--steps llm` — slowest pipeline step.
- First-time model downloads (~2 GB for Moondream).

**GPU acceleration for the GGUF describer** — documented in
`docs/GPU_LLAMA_CPP_CUDA_BUILD.md`. Read that before touching the CUDA toolchain.

**Never modify without explicit instruction:**

- `config/local.yaml` — machine-specific paths.
- Anything under `data/` — user's archive.
- `models/` — downloaded model weights.

---

## Coding conventions

- **Type hints required** on public signatures in `src/kuaa/` and `api/`.
- **Docstrings in English** for public functions.
- **Ruff** for formatting and linting (covers isort). Run before committing.
- **mypy** for type checking.
- **Logging via `logging.getLogger(__name__)`**, never `print()` in production code.
- **Paths via `pathlib.Path`**, never string concatenation.
- **Config via dependency injection** in `api/` (use `api/deps.py`).

For HTML/Jinja:

- Full-page templates in `web/templates/*.html`.
- HTMX fragments in `web/templates/partials/*.html`.
- Element IDs and CSS classes use `kebab-case`.
- All user-visible strings pass through `{{ _("...") }}` for i18n.
