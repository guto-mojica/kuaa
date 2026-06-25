"""C4 — one StatCache; clear_film invalidates exactly one film; counters work."""

from __future__ import annotations

from pathlib import Path

from kuaa.search._cache_core import StatCache, stat_sig


def test_stat_sig_changes_on_write(tmp_path: Path) -> None:
    p = tmp_path / "f.bin"
    p.write_bytes(b"a")
    s1 = stat_sig(p)
    p.write_bytes(b"ab")  # size change
    assert stat_sig(p) != s1


def test_stat_sig_none_for_missing(tmp_path: Path) -> None:
    assert stat_sig(tmp_path / "absent") is None


def test_hit_miss_counters() -> None:
    cache: StatCache[str, int] = StatCache()
    loads = {"n": 0}

    def _load() -> int:
        loads["n"] += 1
        return 42

    assert cache.get_or_load(key="x", signature=(1, 1), loader=_load) == 42  # miss
    assert cache.get_or_load(key="x", signature=(1, 1), loader=_load) == 42  # hit
    assert cache.misses == 1
    assert cache.hits == 1
    assert loads["n"] == 1


def test_clear_film_invalidates_one_slug() -> None:
    cache: StatCache[tuple[str, str], int] = StatCache()
    cache.get_or_load(key=("alpha", "idx"), signature=(1, 1), loader=lambda: 1)
    cache.get_or_load(key=("beta", "idx"), signature=(1, 1), loader=lambda: 2)
    cache.clear_film("alpha")
    hits_before = cache.hits
    # beta still cached (hit), alpha gone (miss → reload).
    cache.get_or_load(key=("beta", "idx"), signature=(1, 1), loader=lambda: 99)
    assert cache.hits == hits_before + 1
