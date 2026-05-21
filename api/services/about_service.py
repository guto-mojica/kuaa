"""About-surface context builder.

Provides the data the redesigned About modal/page renders (Task 29 of the
Mojica frame redesign). The surface has five named blocks:

  * **Stats strip** — 4 cells (films, scenes, runtime, year range).
    Sourced from :func:`cinemateca.library.scan_library` and
    :func:`library_state`; runtime is best-effort (``Film`` does not carry
    a runtime field yet, so the cell falls back to ``"—"``).
  * **Model attributions** — 5 cards (CLIP / Moondream / YOLO / MTCNN /
    CLAP), each with a coloured initial badge, name+version line, role
    description, organisation + GitHub repo link, and a license pill.
    The list is *project-static*: every install ships these models, so the
    builder returns a hard-coded sequence rather than introspecting the
    runtime registry.
  * **Tech stack pills** — coloured chips for the major runtime
    components. Also project-static.
  * **Credits grid** — institutional credit lines (concept / engineering /
    AI integration / funding / year).
  * **Mosaic keyframes** — up to 24 ``/media/...`` URLs harvested from the
    library's per-film ``frames/`` trees, used by the atmospheric
    backdrop. Falls back to an empty list when no library exists yet — the
    backdrop then renders against a flat ``var(--c-bg)``, which is the
    intended "no library yet" look.

The context shape returned by :func:`build_about_context` is documented
inline below and consumed verbatim by
``web/templates/partials/about_modal.html``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from cinemateca import library

# ── Static content (project-wide, not per-request) ───────────────────────────


def model_attributions() -> list[dict[str, Any]]:
    """Return the model-attribution cards rendered in the About modal.

    Order follows the pipeline: visual embedding (CLIP), scene description
    (Moondream), object detection (YOLO), face detection (MTCNN), then
    audio embedding (CLAP). Each entry has:

      * ``key``    — one- or two-char badge text (drives the coloured
                     ``.ab-model .ico`` square at the start of the card).
      * ``color``  — colour variant for the badge: ``""`` (accent purple,
                     the default) / ``"yellow"`` / ``"green"`` / ``"orange"``
                     / ``"pink"``. Maps to the corresponding
                     ``.ab-model .ico.<color>`` rule in ``about.css``.
      * ``name``   — HuggingFace / GitHub identifier, rendered in mono.
      * ``version``— short version tag shown next to the name.
      * ``role``   — short sentence describing what the model does.
      * ``org``    — owning organisation, shown right-aligned.
      * ``lic``    — license string shown in the right-most pill.
      * ``repo_url``— GitHub repo. Optional — when empty/None the "repo"
                     anchor is omitted by the template.
    """
    return [
        {
            "key": "C",
            "color": "",
            "name": "openai/clip-vit-large-patch14",
            "version": "L/14",
            "role": "Visual embedding",
            "org": "OpenAI",
            "lic": "MIT",
            "repo_url": "https://github.com/openai/CLIP",
        },
        {
            "key": "M",
            "color": "yellow",
            "name": "vikhyatk/moondream2",
            "version": "v2",
            "role": "Scene description",
            "org": "Vikhyat",
            "lic": "Apache-2",
            "repo_url": "https://github.com/vikhyat/moondream",
        },
        {
            "key": "Y",
            "color": "green",
            "name": "ultralytics/yolov8m",
            "version": "v8m",
            "role": "Object detection",
            "org": "Ultralytics",
            "lic": "AGPL-3",
            "repo_url": "https://github.com/ultralytics/ultralytics",
        },
        {
            "key": "F",
            "color": "orange",
            "name": "facenet/mtcnn",
            "version": "",
            "role": "Face detection",
            "org": "facenet-pytorch",
            "lic": "MIT",
            "repo_url": "https://github.com/timesler/facenet-pytorch",
        },
        {
            "key": "A",
            "color": "pink",
            "name": "laion/larger_clap_general",
            "version": "general",
            "role": "Audio embedding",
            "org": "LAION",
            "lic": "MIT",
            "repo_url": "https://github.com/LAION-AI/CLAP",
        },
    ]


def tech_stack() -> list[dict[str, Any]]:
    """Return the tech-stack pills shown in the Stack section.

    Each entry has a ``label`` (visible mono text) and an optional
    ``kind`` colour variant: ``""`` (default neutral grey), ``"ac"``,
    ``"green"``, ``"yellow"``, ``"pink"``, or ``"orange"`` — mapped to
    ``.ab-stack-pill.<kind>`` in ``about.css``.
    """
    return [
        {"label": "Python 3.10+", "kind": ""},
        {"label": "FastAPI", "kind": "ac"},
        {"label": "Jinja2", "kind": "ac"},
        {"label": "HTMX 1.9", "kind": "ac"},
        {"label": "PyTorch", "kind": "yellow"},
        {"label": "NumPy", "kind": ""},
        {"label": "FFmpeg", "kind": "pink"},
        {"label": "PySceneDetect", "kind": "green"},
        {"label": "Babel · PT-BR / EN", "kind": ""},
    ]


def credits_list() -> list[dict[str, Any]]:
    """Return the institutional credits grid (label / value pairs).

    ``dim=True`` softens the value to ``--c-text-2`` (used for tertiary
    credits like model authors and acknowledgements).
    """
    return [
        {"role": "Concept", "name": "Cinemateca Mojica · Curatorial team", "dim": False},
        {"role": "Engineering", "name": "Rafael Perez", "dim": False},
        {
            "role": "AI integration",
            "name": "moondream, openai, ultralytics, laion (model authors)",
            "dim": True,
        },
        {"role": "Funding", "name": "—", "dim": True},
        {"role": "Year", "name": "2026", "dim": False},
    ]


# ── Dynamic content (per-request, depends on the library) ─────────────────────


def about_stats(cfg) -> dict[str, Any]:
    """Aggregate library statistics for the 4-cell stats strip.

    Computed lazily from :func:`cinemateca.library.scan_library` —
    cheap (a few JSON reads + a directory scan) but never cached at this
    layer because the library can change between requests (films can be
    added / processed / deleted live).

    Runtime is currently best-effort: ``Film`` does not carry a runtime
    field, so the runtime cell displays ``"—"``. M2/M3 will add per-film
    duration to the registry; this builder will then populate it.
    """
    library_dir = Path(cfg.paths.library_dir)
    films = library.scan_library(library_dir)
    state = library.library_state(library_dir)
    years = sorted({f.year for f in films if f.year})
    if years:
        year_range_display = f"{years[0]}–{years[-1]}" if years[0] != years[-1] else f"{years[0]}"
    else:
        year_range_display = "—"  # em-dash
    return {
        "film_count": len(films),
        "scene_count": state.scene_count,
        "runtime_display": "—",
        "year_range_display": year_range_display,
    }


def mosaic_keyframes(cfg, *, limit: int = 24) -> list[str]:
    """Return up to ``limit`` ``/media/...`` URLs for the atmospheric backdrop.

    Walks every per-film ``<library_dir>/<slug>/frames/`` tree, sorted
    deterministically by slug then path, and emits the first ``limit``
    JPEG URLs it finds. Stops early once ``limit`` is reached.

    The function is purely best-effort: when ``library_dir`` does not
    exist, when no films are registered, or when no keyframes have been
    extracted yet, it returns ``[]`` — the modal then renders against a
    flat ``var(--c-bg)``, which is the intended empty-library look.
    """
    library_dir = Path(cfg.paths.library_dir)
    data_dir = Path(cfg.paths.data_dir).resolve()
    if not library_dir.exists():
        return []

    out: list[str] = []
    for film_dir in sorted(library_dir.iterdir()):
        if not film_dir.is_dir():
            continue
        frames_root = film_dir / "frames"
        if not frames_root.exists():
            continue
        for jpg in sorted(frames_root.rglob("*.jpg")):
            try:
                rel = jpg.resolve().relative_to(data_dir)
            except ValueError:
                # Frame lives outside data_dir (unexpected but defensive):
                # skip rather than emit an unmounted URL.
                continue
            out.append(f"/media/{rel.as_posix()}")
            if len(out) >= limit:
                return out
    return out


# ── Version / build metadata ─────────────────────────────────────────────────


def _read_project_version() -> str:
    """Return the runtime package version, or a sensible fallback."""
    try:
        from cinemateca import __version__ as v

        return v
    except Exception:
        return "1.0.0-beta"


def _read_build_sha() -> str:
    """Return a short git SHA when available, otherwise an empty string.

    The build line in the header renders ``build {sha}`` only when the
    template sees a truthy value, so an empty string collapses the
    field cleanly without conditional Jinja in every caller.
    """
    return ""


def _read_build_date() -> str:
    """Return an ISO-8601 build date (today's date as a best-effort default)."""
    return date.today().isoformat()


# ── Top-level builder ─────────────────────────────────────────────────────────


def build_about_context(cfg) -> dict[str, Any]:
    """Return the full Jinja context for the About modal/page.

    Shape::

        {
            "version":           "0.5.0-beta",
            "build_sha":         "",         # optional
            "build_date":        "2026-05-21",
            "stats":             {film_count, scene_count, runtime_display, year_range_display},
            "models":            [ {key, color, name, version, role, org, lic, repo_url}, ... ],
            "stack":             [ {label, kind}, ... ],
            "credits":           [ {role, name, dim}, ... ],
            "mosaic_keyframes":  [ "/media/...", ... ],   # 0..24 URLs
        }
    """
    return {
        "version": _read_project_version(),
        "build_sha": _read_build_sha(),
        "build_date": _read_build_date(),
        "stats": about_stats(cfg),
        "models": model_attributions(),
        "stack": tech_stack(),
        "credits": credits_list(),
        "mosaic_keyframes": mosaic_keyframes(cfg),
    }
