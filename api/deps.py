"""FastAPI dependency providers."""
from functools import cache, lru_cache
from pathlib import Path

from fastapi import Request

_LOCALES_DIR = Path(__file__).parent.parent / "web" / "locales"
_SUPPORTED_LOCALES = {"pt_BR", "en"}

# Maps an internal locale code to a BCP-47 value for <html lang="...">.
# Default falls back to the project default locale (pt_BR).
_LOCALE_LANG = {"pt_BR": "pt-BR", "en": "en"}
_DEFAULT_LANG = "pt-BR"


def locale_to_lang(locale: str) -> str:
    """Return the BCP-47 ``lang`` attribute value for a locale code."""
    return _LOCALE_LANG.get(locale, _DEFAULT_LANG)


@lru_cache(maxsize=1)
def get_config():
    from cinemateca.config import load_config

    local = Path("config/local.yaml")
    return load_config(str(local) if local.exists() else None)


@cache
def _get_translations(locale: str):
    from babel.support import Translations

    if locale not in _SUPPORTED_LOCALES:
        locale = "pt_BR"
    return Translations.load(str(_LOCALES_DIR), [locale])


def make_ctx(request: Request, **kwargs) -> dict:
    """Build a Jinja2 template context with per-request locale and active film."""
    locale = request.cookies.get("locale", "pt_BR")
    trans = _get_translations(locale)
    return {
        "request": request,
        "_": trans.gettext,
        "locale": locale,
        "lang": locale_to_lang(locale),
        "active_film": request.cookies.get("active_film", ""),
        **kwargs,
    }


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
