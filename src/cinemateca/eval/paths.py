"""Filesystem path resolution for the eval pipeline.

Reads cfg.eval.root and cfg.eval.run_id (or their defaults) and returns
concrete Path / str values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def eval_root(cfg: Any) -> Path:
    """Resolve the eval data root from config, falling back to ``data/eval``.

    Tests ``monkeypatch.setattr`` this function to redirect writes to
    a tmp dir. The runtime path goes through ``cfg.eval.root`` (new
    config block added in Task 30) when present; otherwise it derives
    a path under ``cfg.paths.data_dir`` to stay inside the project
    sandbox; otherwise the literal ``"data/eval"`` fallback.
    """

    eval_cfg = getattr(cfg, "eval", None)
    if eval_cfg is not None:
        root = getattr(eval_cfg, "root", None)
        if root:
            return Path(root)
    paths = getattr(cfg, "paths", None)
    if paths is not None:
        data_dir = getattr(paths, "data_dir", None)
        if data_dir:
            return Path(data_dir) / "eval"
    return Path("data/eval")


def eval_run_id(cfg: Any) -> str:
    """Resolve the current run id from config, falling back to ``"default"``."""

    eval_cfg = getattr(cfg, "eval", None)
    if eval_cfg is not None:
        run_id = getattr(eval_cfg, "run_id", None)
        if run_id:
            return str(run_id)
    return "default"
