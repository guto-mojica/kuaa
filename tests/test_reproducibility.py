"""Deterministic seeding utility (F3)."""

from __future__ import annotations

import numpy as np

from cinemateca.reproducibility import make_generator, seed_everything


def test_seed_everything_makes_python_and_numpy_deterministic():
    import random

    seed_everything(123)
    a_py = [random.random() for _ in range(3)]
    a_np = np.random.rand(3).tolist()
    seed_everything(123)
    b_py = [random.random() for _ in range(3)]
    b_np = np.random.rand(3).tolist()
    assert a_py == b_py
    assert a_np == b_np


def test_seed_everything_seeds_torch_if_present():
    seed_everything(7)
    try:
        import torch
    except ImportError:
        return
    x = torch.rand(4)
    seed_everything(7)
    y = torch.rand(4)
    assert torch.equal(x, y)


def test_make_generator_is_pure_and_repeatable():
    g1 = make_generator(42, "anchor", 351, 412)
    g2 = make_generator(42, "anchor", 351, 412)
    assert g1.random() == g2.random()
    # Different salt → different stream.
    assert (
        make_generator(42, "anchor", 351, 412).random()
        != make_generator(42, "anchor", 351, 999).random()
    )
