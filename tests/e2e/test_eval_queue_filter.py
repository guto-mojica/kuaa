"""Eval queue — the "No queries match this filter." state.

``noMatches()`` lives inline in ``web/templates/eval/queue.html`` as Alpine
state, so nothing but a browser can exercise it. It must appear exactly when
the run has queue rows and the active filter hides every one of them — and
not for an empty run, which has its own message.

Read-only: the tests type into the filter box and click a tab. No grade is
posted, so ``data/eval`` is untouched. Skips when this machine has no eval run
seeded, since the queue is then empty and there is nothing to filter.
"""

from __future__ import annotations

from typing import Any

import pytest

from .conftest import EVAL_TOKEN

pytestmark = pytest.mark.e2e

_ROW = ".ev-q-list .ev-q"
# The filter-empty message is the ``.ev-q-empty`` bound to ``x-show``; the
# no-queries message beside it is static. Selected by structure, not text,
# because the served locale decides the wording.
_NO_MATCH = ".ev-q-list .ev-q-empty[x-show]"


def _open_queue(page: Any) -> int:
    """Open the eval pane on the All tab; return the queue row count or skip."""
    page.goto(f"/eval?token={EVAL_TOKEN}&filter=todas")
    page.wait_for_selector(".ev-lp")
    count = page.locator(_ROW).count()
    if count == 0:
        pytest.skip("no eval run seeded on this machine — queue is empty")
    return count


def test_no_match_message_follows_the_search_box(page: Any) -> None:
    count = _open_queue(page)
    no_match = page.locator(_NO_MATCH)
    assert not no_match.is_visible(), "message must be hidden while rows match"

    page.fill(".ev-lp .filter input", "zzz-no-query-says-this-zzz")
    no_match.wait_for(state="visible")
    assert page.locator(f"{_ROW}:visible").count() == 0

    page.fill(".ev-lp .filter input", "")
    no_match.wait_for(state="hidden")
    assert page.locator(f"{_ROW}:visible").count() == count


def test_pending_tab_keeps_the_message_hidden_for_a_fresh_grader(page: Any) -> None:
    """A fresh browser context has no grader cookie, so every row is pending
    for it: the Pending tab shows the whole queue and the message stays
    hidden. The all-hidden case is covered by the search-box test above; it
    is not reachable here without a grader who has judged the entire run."""
    count = _open_queue(page)
    assert page.locator(f"{_ROW}[data-status='pending']").count() == count
    page.click(".lpf-tabs .t[data-filter='pendentes']")
    assert not page.locator(_NO_MATCH).is_visible()
    assert page.locator(f"{_ROW}:visible").count() == count
