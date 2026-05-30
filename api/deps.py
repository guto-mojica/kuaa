"""FastAPI dependency providers."""

import json
import os
from functools import cache, lru_cache
from pathlib import Path
from typing import Literal

from fastapi import Depends, Query, Request, Response

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
    """Extract the ``?film=<slug>`` query param, validated against the library.

    Returns ``None`` when the parameter is absent, empty, or names a slug
    whose per-film directory does not exist on disk — meaning "aggregate
    across all registered films". Returns the validated slug string only
    when present, non-empty, AND the directory exists.

    **Silent-aggregate fallback is intentional and must not be changed to
    raise.** Every ``/tab/*`` and ``/api/*`` fragment route depends on this
    dependency. When a user has a stale ``active_film`` cookie (e.g. the film
    was deleted, or the slug was renamed) the cookie propagates into every
    HTMX fragment request via ``?film=<slug>``. If this dependency raised
    ``IndexMissing`` or any other exception on an unknown slug, every
    fragment route would 5xx and the entire UI would break until the user
    manually cleared the cookie. The aggregate view is a safe, graceful
    fallback: the user still sees results, and can select a different film
    from the sidebar. This contract is pinned by
    ``test_tab_scenes_unknown_slug_falls_back_to_aggregate`` and explicitly
    documented here so future refactors don't silently regress it.

    Contrast with ``api_library_select`` (``/api/library/select/{slug}``),
    which is an explicit navigation intent and raises ``IndexMissing`` (404)
    on an unknown slug — that is the correct place to surface the error.
    """
    if not film:
        return None
    from pathlib import Path

    slug = film.lower()
    cfg = get_config()
    film_dir = Path(cfg.paths.library_dir) / slug
    return slug if film_dir.is_dir() else None


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
        # Chrome defaults — active_job_count still drives the Processing badge.
        # Collaboration identity/notification keys are retained for future
        # surfaces but are not rendered by the launch topbar.
        "active_job_count": 0,
        "viewers": [],
        "notification_count": 0,
        "current_user": None,
        # Mojica redesign (Task 10+): the Buscar tab reads retrieval UI
        # gates/defaults straight from ``cfg.search.*``. Exposing the
        # full config here keeps the routes simple and avoids a separate
        # dependency for templates.
        "cfg": get_config(),
        # Profile-aware reranker default (GPU-on / CPU-off when
        # ``retrieval.reranker.enabled: auto``). The Buscar store seeds its
        # Rerank toggle with this on a browser with no saved preference;
        # localStorage + ``?reranker_enabled=`` still override. Lazy import
        # avoids an api.deps ↔ api.services.search cycle.
        "reranker_default": _reranker_default_enabled(),
    }
    base.update(kwargs)
    return base


def _reranker_default_enabled() -> bool:
    """Profile-resolved reranker default for the page shell (see base ctx)."""
    from api.services.search import reranker_default_enabled

    return reranker_default_enabled(get_config())


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

    from cinemateca.library import FilmContext

    if cfg is None:
        cfg = get_config()
    slug = request.cookies.get("active_film", "").lower()
    if slug:
        film_dir = Path(cfg.paths.library_dir) / slug
        if film_dir.exists():
            return FilmContext.for_film(cfg, slug)
    return FilmContext.from_config(cfg)


# ── A6: FilmContext FastAPI dependencies ──────────────────────────────────────


def resolve_film_context(cfg, slug: str | None, request):
    """Resolve a FilmContext outside FastAPI DI (e.g. ``render_page``).

    Precedence: explicit ``slug`` → cookie on ``request`` → flat aggregate.
    Pass ``request=None`` when the caller always supplies a slug (cookie
    fallback skipped).
    """
    from cinemateca.library import FilmContext

    if slug is not None:
        return FilmContext.for_film(cfg, slug)
    if request is not None:
        return film_ctx(request, cfg)
    return FilmContext.from_config(cfg)


def flat_film_context():
    """Return a FilmContext across all films (no per-film scope).

    Intended as a plain helper (not a Depends) for callers that need the
    aggregate context when ``optional_film_context`` returns ``None``.
    """
    from cinemateca.library import FilmContext

    return FilmContext.from_config(get_config())


def optional_film_context(request: Request, slug: str | None = Depends(film_slug_query)):
    """Resolve the active FilmContext, or None for the aggregate view.

    ``slug`` already validated by ``film_slug_query`` (existing dir or
    None). ``None`` → aggregate across the library; the caller branches
    on it.
    """
    from cinemateca.library import FilmContext

    cfg = get_config()
    if slug is None:
        return None
    return FilmContext.for_film(cfg, slug)


def required_film_context(slug: str | None = Depends(film_slug_query)):
    """Like ``optional_film_context`` but 404s when no film resolves."""
    from cinemateca.errors import IndexMissing
    from cinemateca.library import FilmContext

    cfg = get_config()
    if slug is None:
        raise IndexMissing("a film slug is required for this endpoint")
    return FilmContext.for_film(cfg, slug)


def annotate_film_context(request: Request, slug: str | None = Depends(film_slug_query)):
    """Resolve the FilmContext for annotate routes.

    Precedence: ``?film=<slug>`` → ``active_film`` cookie → flat aggregate.
    Mirrors :func:`api.services.annotations.resolve_film_context` so
    ``render_page`` and annotate routes share the same logic via
    :func:`resolve_film_context`.
    """
    return resolve_film_context(get_config(), slug, request)
