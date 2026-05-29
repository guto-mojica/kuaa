"""Global + scoped deterministic seeding (F3).

``seed_everything`` pins the process-wide PRNGs (python ``random``,
numpy legacy global, and torch CPU+CUDA if installed) at pipeline and
eval start so a fixed config seed makes a run reproducible. ``torch`` is
imported lazily so this module stays importable in torch-free contexts
(docs builds, lint).

``make_generator`` returns an independent, salted
:class:`numpy.random.Generator` for code that needs *local* determinism
without mutating global state (replaces the ad-hoc ``sum(ord(c))`` PRNG
in ``rhymes/enrich.py``).
"""
from __future__ import annotations

import logging
import os
import random
import zlib
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed python ``random``, numpy, and torch (if importable).

    Also sets ``PYTHONHASHSEED`` for child processes; the current
    interpreter's hash seed is already fixed at startup, so this only
    affects subprocesses (documented limitation).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        logger.debug("torch not installed; skipping torch seeding")


def make_generator(seed: int, *salt: Any) -> np.random.Generator:
    """Return a deterministic, salted numpy Generator.

    The salt (e.g. anchor + echo scene ids) is folded into the seed via
    CRC32 so distinct call sites get independent, repeatable streams
    without touching the global PRNG.
    """
    key = "|".join(str(s) for s in (seed, *salt)).encode("utf-8")
    return np.random.default_rng(zlib.crc32(key))
