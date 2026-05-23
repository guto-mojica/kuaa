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


def get_library_dir() -> Path:
    """Return the configured library directory."""
    return Path(get_config().paths.library_dir)



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
    """Build a Jinja2 template context with per-request locale and current film."""
    locale = request.cookies.get("locale", "pt_BR")
    trans = _get_translations(locale)
    return {
        "request": request,
        "_": trans.gettext,
        "locale": locale,
        "lang": locale_to_lang(locale),
        **kwargs,
    }
