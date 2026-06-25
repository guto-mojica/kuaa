"""Timing hook (F5)."""

from __future__ import annotations

import time

from kuaa.timing import Timer, timed


def test_timed_context_manager_measures_elapsed():
    with timed() as t:
        time.sleep(0.01)
    assert isinstance(t, Timer)
    assert t.elapsed_ms >= 10.0
    assert t.elapsed_ms < 1000.0


def test_timed_context_records_label_and_logs(caplog):
    import logging

    with caplog.at_level(logging.DEBUG, logger="kuaa.timing"):
        with timed("search.encode") as t:
            time.sleep(0.001)
    assert t.label == "search.encode"
    assert any("search.encode" in r.message for r in caplog.records)


def test_timer_elapsed_ms_stable_after_exit():
    with timed() as t:
        time.sleep(0.001)
    first = t.elapsed_ms
    time.sleep(0.005)
    assert t.elapsed_ms == first  # frozen at __exit__
