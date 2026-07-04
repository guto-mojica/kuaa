"""Pre-processing tab — context builder + start-response composer.

The Pre-processing surface drives ``api.jobs.PREPROCESS_STEPS``-only pipeline
runs (scene detection — a root in the dependency graph, so nothing
downstream depends on running it here) and a filmstrip cut-review UI. It
reuses the Processing tab's job registry and SSE stream
(``build_processing_context``, ``/api/pipeline/stream``) so a Pre-processing
run is still tracked by the same single-active-job policy, but renders its
own compact job card (``preprocess_job.html`` / ``preprocess_stepper.html``)
rather than the Processing tab's full one — see ``JobState.is_preprocess_only``.
This module adds the per-film filmstrip view model
(``kuaa.preprocess.build_filmstrip``) on top.
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


def film_video_path(cfg, slug: str) -> Path | None:
    """Return the raw video file path for a registered slug, or ``None``."""
    for film in scan_library(Path(cfg.paths.library_dir)):
        if film.slug == slug:
            return film.raw_path if film.raw_path.exists() else None
    return None


def _filmstrip_for(cfg, slug: str | None) -> tuple[dict[str, Any] | None, str, str]:
    """Build the filmstrip view model, film title, and raw-video URL for ``slug``.

    Returns ``(None, "", "")`` when no film is selected or it is not registered.
    The raw-video URL feeds the review player and is served off the ``/media``
    mount (range-request seeking supported), so scrubbing stays fully offline.
    """
    if not slug:
        return None, "", ""
    from kuaa.preprocess import build_filmstrip

    try:
        ctx = FilmContext.for_film(cfg, slug)
    except ValueError:
        return None, "", ""
    title = next(
        (f.title for f in scan_library(Path(cfg.paths.library_dir)) if f.slug == slug), slug
    )
    # Serve the raw video through a dedicated range-capable route rather than the
    # /media mount: the source often lives outside data_dir (symlinked archive),
    # which the static mount cannot resolve. Empty when no source is on disk —
    # the template then omits the player and explains editing needs the source.
    _ = ctx  # data_dir no longer used for the video URL; kept for clarity
    video_url = f"/api/preprocess/video/{slug}" if film_video_path(cfg, slug) else ""
    return build_filmstrip(ctx), title, video_url


def build_preprocess_context(slug: str | None) -> dict[str, Any]:
    """Build the template context for the Pre-processing tab.

    Reuses ``build_processing_context`` (scoped to scene-detection jobs) for the
    film selector, any active detection job, and the SSE log seed, then layers
    the selected film's filmstrip + review player on top.
    """
    cfg = get_config()
    base: dict[str, Any] = dict(build_processing_context(surface="preprocess"))
    filmstrip, title, video_url = _filmstrip_for(cfg, slug)
    base.update(
        {
            "filmstrip": filmstrip,
            "pp_slug": slug or "",
            "pp_title": title,
            "pp_video_url": video_url,
            "pp_fps": filmstrip["fps"] if filmstrip else 24.0,
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
    filmstrip, title, _video = _filmstrip_for(cfg, slug)
    return templates.TemplateResponse(
        request,
        "partials/preprocess_filmstrip.html",
        make_ctx(request, filmstrip=filmstrip, pp_slug=slug, pp_title=title),
    )
