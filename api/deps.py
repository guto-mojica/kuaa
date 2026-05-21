"""FastAPI dependency providers."""
import os
from functools import cache, lru_cache
from pathlib import Path

from fastapi import Query, Request

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
    ``compact_lp``, ``has_right_pane``, ``breadcrumb``, ``page_title``) so the
    new shell renders sensibly even when a route forgets to set them. Callers
    that pass any of these via ``**kwargs`` override the defaults.
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
    }
    base.update(kwargs)
    return base


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
