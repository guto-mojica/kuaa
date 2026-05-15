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
- No rewriting `src/cinemateca/*` during the migration — it's the working core.

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
| Cenas | Scenes | Library browsing tab (was "Catálogo" in v0.2.x) |
| Anotar | Annotate | Manual tag curation tab |
| Processamento | Processing | Tab visible only when active jobs exist |
| Sobre | About | Institutional credits modal/page |
| Tag | Tag | Label applicable to a scene (automatic or manual) |
| Pipeline | Pipeline | Step sequence: frames → scenes → visual → embeddings → llm |

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
docs/                DESIGN_SYSTEM.md, ARCHITECTURE.md, MIGRATION_NOTES.md.
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
# One-time setup
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[full,dev]"

# Run the FastAPI interface (v0.3.0+)
python app.py
# Opens at http://localhost:8501

# Run the legacy Streamlit interface (during migration)
streamlit run app_streamlit.py

# Tests
pytest tests/ -q
pytest tests/test_smoke.py -v   # smoke only, no heavy models

# CLI
python -m cinemateca info --video data/raw/jeca_tatu.mp4
python -m cinemateca process --video data/raw/jeca_tatu.mp4
python -m cinemateca process --video data/raw/jeca_tatu.mp4 --steps scenes,embeddings

# i18n
pybabel extract -F web/babel.cfg -o web/locales/messages.pot web/
pybabel update -i web/locales/messages.pot -d web/locales/
pybabel compile -d web/locales/
```

---

## Constraints and care

**Expensive operations — require explicit user confirmation before running:**

- `python -m cinemateca process` on any video (~60–120min on CPU).
- Embedding regeneration (`--steps embeddings`) for the full library.
- LLM description regeneration (`--steps llm`) — slowest pipeline step.
- First-time model downloads (~2GB for Moondream).

For quick development checks, use the test film **Jeca Tatu (1959)** already
processed in `data/`. Artefacts persist across runs.

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
- **Black + isort** (configured in `pyproject.toml`). Run before committing.
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
- [x] Streamlit parity confirmed → tagged `v0.2.1-streamlit-final` at a373fd7

Keep this list updated as steps complete.

---

*This document is alive. When you notice it's outdated or imprecise, propose an edit.*
