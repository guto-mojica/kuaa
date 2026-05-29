"""cinemateca.config — typed settings (F1).

Public surface unchanged for callers: ``load_config``, ``setup_logging``,
the ``Config`` alias, plus dot-access / ``.get`` / ``.to_dict`` on the
returned model. ``config.py`` became this package; the schema lives in
:mod:`cinemateca.config.schema`, the loader in
:mod:`cinemateca.config.loader`.
"""

from __future__ import annotations

from cinemateca.config.loader import load_config, setup_logging
from cinemateca.config.schema import CONFIG_VERSION, Settings

Config = Settings  # back-compat alias; was ``Config = _Namespace``.

# Back-compat private alias: ``scene_detector`` and ``data_prep`` annotate
# ``cfg: _Namespace | None`` under ``TYPE_CHECKING`` and import this name.
# The old ``_Namespace`` class is gone; the typed root is its successor.
_Namespace = Settings

__all__ = ["CONFIG_VERSION", "Config", "Settings", "load_config", "setup_logging"]
