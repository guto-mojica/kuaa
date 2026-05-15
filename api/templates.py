"""Shared Jinja2 templates instance. Import this instead of creating a new one."""
from pathlib import Path

from fastapi.templating import Jinja2Templates

_BASE = Path(__file__).parent.parent
templates = Jinja2Templates(directory=str(_BASE / "web" / "templates"))
# Placeholder until Babel translations are wired (task: i18n PT/EN)
templates.env.globals["_"] = lambda s, **kw: s
