"""FastAPI dependency providers."""

import json
import os
from functools import cache, lru_cache
from pathlib import Path
from typing import Literal

from fastapi import Query, Request, Response

ToastKind = Literal["info", "success", "warn", "error"]

CONFIG_ENV_VAR = "CINEMATECA_CONFIG"

_LOCALES_DIR = Path(__file__).parent.parent / "web" / "locales"
_SUPPORTED_LOCALES = {"pt_BR", "en"}

# Maps an internal locale code to a BCP-47 value for <html lang="...">.
# Default falls back to the project default locale (pt_BR).
_LOCALE_LANG = {"pt_BR": "pt-BR", "en": "en"}
_DEFAULT_LANG = "pt-BR"


def locale_to_lang(locale: str) -> str:
    """Return the BCP-47 ``lang`` attribute value for a locale code."""
    return _LOCALE_LANG.get(locale, _DEFAULT_LANG)


def selected_config_path() -> Path | None:
    """Return the config override path, preserving the historical fallback.

    Precedence is:

    1. ``CINEMATECA_CONFIG`` environment variable, set by ``app.py --config``.
    2. ``config/local.yaml`` when present.
    3. ``None`` so ``cinemateca.config.load_config`` uses defaults only.
    """
    explicit = os.environ.get(CONFIG_ENV_VAR)
    if explicit:
        return Path(explicit).expanduser()

    local = Path("config/local.yaml")
    return local if local.exists() else None


@lru_cache(maxsize=1)
def get_config():
    from cinemateca.config import load_config

    selected = selected_config_path()
    return load_config(str(selected) if selected is not None else None)


@cache
def _get_translations(locale: str):
    from babel.support import Translations

    if locale not in _SUPPORTED_LOCALES:
        locale = "pt_BR"
    return Translations.load(str(_LOCALES_DIR), [locale])


def film_slug_query(
    film: str | None = Query(
        default=None,
        description="Slug filter; omit for aggregate view",
    ),
) -> str | None:
    """Extract the ``?film=<slug>`` query param.

    Returns ``None`` when the parameter is absent, meaning "aggregate
    across all registered films". Returns the slug string when present.
    """
    return film


def make_ctx(request: Request, **kwargs) -> dict:
    """Build a Jinja2 template context with per-request locale and active film.

    Also defaults the Mojica chrome context keys (``active_tab``,
    ``compact_lp``, ``has_right_pane``, ``breadcrumb``, ``page_title``,
    ``active_job_count``, ``viewers``, ``notification_count``,
    ``current_user``) so the new shell renders sensibly even when a route
    forgets to set them. Callers that pass any of these via ``**kwargs``
    override the defaults.

    The merged effective config is injected as ``cfg`` so templates can
    read read-only knobs (``cfg.search.hybrid_sem_w`` etc.) without each
    route having to pass it explicitly. This is the same Namespace
    returned by ``get_config()``; it is cached, so the lookup is cheap.
    """
    locale = request.cookies.get("locale", "pt_BR")
    trans = _get_translations(locale)
    base = {
        "request": request,
        "_": trans.gettext,
        "locale": locale,
        "lang": locale_to_lang(locale),
        "active_film": request.cookies.get("active_film", ""),
        # Mojica chrome defaults — overridden by render_page() per route.
        "active_tab": "search",
        "compact_lp": False,
        "has_right_pane": True,
        "breadcrumb": [],
        "page_title": None,
        # TopBar (Task 7) defaults — Task 8's chrome_service will replace
        # these with the real per-request values (active job count derived
        # from the jobs registry, viewers from the collaboration epic, etc.).
        # For now the topbar renders with a 0-count tab badge, no viewers
        # stack, and an anonymous "M" avatar when these aren't supplied.
        "active_job_count": 0,
        "viewers": [],
        "notification_count": 0,
        "current_user": None,
        # Mojica redesign (Task 10): the Buscar tab reads display-only
        # retrieval knobs straight from ``cfg.search.*``. Exposing the
        # full config here keeps the routes simple and avoids a separate
        # dependency for templates.
        "cfg": get_config(),
    }
    base.update(kwargs)
    return base


def toast_trigger(
    response: Response,
    *,
    title: str,
    sub: str | None = None,
    kind: ToastKind = "info",
    duration: int | None = None,
) -> None:
    """Attach an ``HX-Trigger`` header so the client pushes a toast.

    Phase 7 / Task 26 ships the client-side ToastBus (see
    ``web/static/js/mojica.js``). Any route can call this helper to
    surface a notification on the next HTMX response:

    .. code-block:: python

        @router.post("/api/things/save")
        async def save(response: Response, ...):
            ...  # do the work
            toast_trigger(response, title="Saved", kind="success")
            return templates.TemplateResponse(...)

    The header value is a JSON object whose ``toast`` key carries the
    spec consumed by ``window.ToastBus.push(spec)``::

        HX-Trigger: {"toast": {"title":"Saved","kind":"success"}}

    htmx dispatches a CustomEvent named ``toast`` with the inner object
    as ``evt.detail``; the bus listens on ``document.body``.

    Parameters
    ----------
    response:
        The FastAPI ``Response`` (injected by FastAPI when the route
        signature declares it). Headers are mutated in-place.
    title:
        Required top line.
    sub:
        Optional second line (small caption under the title).
    kind:
        Visual variant: ``info`` (default), ``success``, ``warn``,
        ``error``. Drives the left bar colour and icon tint.
    duration:
        Auto-dismiss in ms. ``None`` keeps the client default (3500ms).
        Pass ``0`` to disable auto-dismiss (the user must click the
        close button).

    Notes
    -----
    Calling this helper twice on the same response overwrites the prior
    header — htmx accepts a single ``HX-Trigger`` value per response.
    If a route needs to fire multiple toasts in a single response, batch
    them with a custom event key (out of scope for Task 26).
    """
    payload: dict[str, object] = {"title": title, "kind": kind}
    if sub:
        payload["sub"] = sub
    if duration is not None:
        payload["duration"] = duration
    response.headers["HX-Trigger"] = json.dumps({"toast": payload})


def film_ctx(request: Request, cfg=None):
    """Return a FilmContext for the currently active film (from cookie).

    Falls back to the global flat context when no film cookie is set or the
    per-film directory does not exist yet.
    """
    from pathlib import Path

    from api.services.film_context import FilmContext

    if cfg is None:
        cfg = get_config()
    slug = request.cookies.get("active_film", "")
    if slug:
        film_dir = Path(cfg.paths.data_dir).resolve() / "films" / slug
        if film_dir.exists():
            return FilmContext.for_film(cfg, slug)
    return FilmContext.from_config(cfg)
