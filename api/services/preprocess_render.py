"""Pre-processing tab — context builder + start-response composer.

The Pre-processing surface drives scene-detection-only pipeline runs and a
filmstrip cut-review UI. It deliberately reuses the Processing tab's job
registry, SSE stream, and job-card machinery (``build_processing_context``,
``/api/pipeline/stream``); this module only adds the per-film filmstrip view
model (``kuaa.preprocess.build_filmstrip``) on top.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from api.deps import get_config, make_ctx
from api.services.processing_render import _derive_slug, build_processing_context
from api.templates import templates
from kuaa.library import FilmContext, scan_library

logger = logging.getLogger(__name__)

# Only scene detection runs from the Pre-processing surface.
PREPROCESS_STEPS: set[str] = {"scene_detection"}


def film_video_path(cfg, slug: str) -> Path | None:
    """Return the raw video file path for a registered slug, or ``None``."""
    for film in scan_library(Path(cfg.paths.library_dir)):
        if film.slug == slug:
            return film.raw_path if film.raw_path.exists() else None
    return None


def _filmstrip_for(cfg, slug: str | None) -> tuple[dict[str, Any] | None, str]:
    """Build the filmstrip view model + film title for ``slug``.

    Returns ``(None, "")`` when no film is selected or it is not registered.
    """
    if not slug:
        return None, ""
    from kuaa.preprocess import build_filmstrip

    try:
        ctx = FilmContext.for_film(cfg, slug)
    except ValueError:
        return None, ""
    title = next(
        (f.title for f in scan_library(Path(cfg.paths.library_dir)) if f.slug == slug), slug
    )
    return build_filmstrip(ctx), title


def build_preprocess_context(slug: str | None) -> dict[str, Any]:
    """Build the template context for the Pre-processing tab.

    Reuses ``build_processing_context`` for the film selector, any active job,
    and the SSE log seed, then layers the selected film's filmstrip on top.
    """
    cfg = get_config()
    base: dict[str, Any] = dict(build_processing_context())
    filmstrip, title = _filmstrip_for(cfg, slug)
    base.update(
        {
            "filmstrip": filmstrip,
            "pp_slug": slug or "",
            "pp_title": title,
        }
    )
    return base


def build_preprocess_start_response(request, cfg, video_path: Path, cookie_slug: str):
    """Compose the HTML response after a scene-detection run is accepted.

    Mirrors ``build_start_response`` but re-renders the Pre-processing tab so
    the running job's SSE log + stepper mount in place, and sets the
    ``active_film`` cookie to the detected slug.
    """
    from fastapi.responses import HTMLResponse

    from api.services.chrome_service import build_chrome_context

    new_slug = _derive_slug(cfg, video_path, cookie_slug)

    ctx = build_preprocess_context(new_slug)
    tab_html = templates.env.get_template("partials/preprocess.html").render(
        make_ctx(request, active_film=new_slug, current_slug=new_slug, **ctx)
    )

    chrome_ctx = build_chrome_context(cfg, current_slug=new_slug)
    lp_payload: dict = dict(chrome_ctx)
    lp_payload.update({"active_film": new_slug, "current_slug": new_slug})
    lp_html = templates.env.get_template("partials/_left_pane_body.html").render(
        make_ctx(request, **lp_payload)
    )
    oob = f'<div id="lp-scroll" hx-swap-oob="innerHTML">{lp_html}</div>'

    response = HTMLResponse(tab_html + oob)
    response.set_cookie(
        "active_film", new_slug, max_age=86400 * 365, httponly=False, samesite="lax"
    )
    return response


def render_filmstrip_fragment(request, slug: str):
    """Render just the filmstrip fragment for ``slug`` (post-detect refresh)."""
    cfg = get_config()
    filmstrip, title = _filmstrip_for(cfg, slug)
    return templates.TemplateResponse(
        request,
        "partials/preprocess_filmstrip.html",
        make_ctx(request, filmstrip=filmstrip, pp_slug=slug, pp_title=title),
    )
