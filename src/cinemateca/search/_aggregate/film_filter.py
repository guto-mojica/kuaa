"""Registry walk + valid-index gate for the aggregate pipeline (C1).

Collapses the pre-C1 *double* index load — the pre-scan loop that built
``valid_slugs`` and the main loop that re-loaded each index — into a
single pass. :class:`FilmFilter` loads each film's index exactly ONCE
and hands the orchestrator the loaded :class:`SearchIndex` on a
:class:`CandidateFilm`, so the scoring phase reads ``idx.embeddings`` /
``idx.kf_df`` straight off the candidate and never re-loads.

The skip logging (``ValueError`` → warning, non-OK status → info) is
preserved verbatim from the pre-C1 main loop; the pre-scan loop emitted
no logs, so collapsing the two passes does not add or drop any log line
for a skipped film.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from cinemateca.search.cache import IndexStatus, SearchIndex

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateFilm:
    """A registered film whose CLIP index loaded OK, with that index attached."""

    slug: str
    index: SearchIndex


class FilmFilter:
    """Walk slugs, load each index ONCE, keep the ``IndexStatus.OK`` ones.

    ``load_index`` is injectable so unit tests can count loads; it
    defaults to :func:`cinemateca.search.aggregate._get_search_index`
    (resolved lazily to avoid an import cycle), the cached production
    loader.
    """

    def __init__(self, load_index: Callable[[Any, str], SearchIndex] | None = None) -> None:
        self._load_index = load_index

    def _loader(self) -> Callable[[Any, str], SearchIndex]:
        if self._load_index is not None:
            return self._load_index
        from cinemateca.search.aggregate import _get_search_index

        return _get_search_index

    def candidates(self, *, cfg: Any, slugs: list[str]) -> list[CandidateFilm]:
        """Return one :class:`CandidateFilm` per slug whose index is OK.

        Loads each slug's index exactly once. A slug whose loader raises
        ``ValueError`` (unregistered / invalid) is skipped with a warning;
        a slug whose index status is not ``OK`` is skipped with an info
        line — both verbatim from the pre-C1 main loop.
        """
        load_index = self._loader()
        candidates: list[CandidateFilm] = []
        for slug in slugs:
            try:
                idx = load_index(cfg, slug)
            except ValueError as exc:
                logger.warning("aggregate_search: skip film %s — %s", slug, exc)
                continue
            if idx.status is not IndexStatus.OK:
                logger.info(
                    "aggregate_search: skip film %s — index status %s",
                    slug,
                    idx.status,
                )
                continue
            candidates.append(CandidateFilm(slug=slug, index=idx))
        return candidates
