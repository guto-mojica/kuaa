"""FastAPI dependency providers."""
from functools import lru_cache
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


@lru_cache(maxsize=None)
def _get_translations(locale: str):
    from babel.support import Translations

    if locale not in _SUPPORTED_LOCALES:
        locale = "pt_BR"
    return Translations.load(str(_LOCALES_DIR), [locale])


def make_ctx(request: Request, **kwargs) -> dict:
    """Build a Jinja2 template context with per-request locale."""
    locale = request.cookies.get("locale", "pt_BR")
    trans = _get_translations(locale)
    return {
        "request": request,
        "_": trans.gettext,
        "locale": locale,
        "lang": locale_to_lang(locale),
        **kwargs,
    }
