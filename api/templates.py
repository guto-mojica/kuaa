"""Shared Jinja2 templates instance. Import this instead of creating a new one."""
from pathlib import Path

from babel.support import Translations
from fastapi.templating import Jinja2Templates

_BASE = Path(__file__).parent.parent
_LOCALES_DIR = _BASE / "web" / "locales"

templates = Jinja2Templates(directory=str(_BASE / "web" / "templates"))

# Default to pt_BR for SSE and other non-request rendering contexts.
# Per-request locale overrides this via make_ctx() in api/deps.py.
_pt_br = Translations.load(str(_LOCALES_DIR), ["pt_BR"])
templates.env.globals["_"] = _pt_br.gettext
