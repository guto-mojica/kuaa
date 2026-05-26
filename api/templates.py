"""Shared Jinja2 templates instance. Import this instead of creating a new one."""

import subprocess
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

# Cache-busting token for static assets: git SHA or fallback timestamp.
# Appended as ?v=<token> to CSS URLs in base.html so browsers pick up
# changes without a manual hard-refresh.
try:
    _css_version = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(_BASE), text=True,
    ).strip()[:7]
except Exception:
    from time import time
    _css_version = str(int(time()))
templates.env.globals["css_version"] = _css_version
