"""FastAPI dependency providers."""
from functools import lru_cache
from pathlib import Path

from fastapi import Request

_LOCALES_DIR = Path(__file__).parent.parent / "web" / "locales"
_SUPPORTED_LOCALES = {"pt_BR", "en"}


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
    return {"request": request, "_": trans.gettext, "locale": locale, **kwargs}
