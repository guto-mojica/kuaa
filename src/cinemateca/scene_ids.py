"""
cinemateca.scene_ids
~~~~~~~~~~~~~~~~~~~~~
Canonical scene-ID representation for tag filtering.

Why this module exists
----------------------
Scene IDs enter the system from two sources with two different types:

  * ``LLMDescriber.build_tag_index`` stores ``scene_id`` as **int**
    (records use ``int(row.get("scene_id", -1))``). Saved/loaded via JSON
    these stay Python ints because they are list *values*, not object keys.
  * Manual annotations (``annotator.load``) are a JSON *object*, so their
    keys are **strings**. ``annotator.merge_tag_index`` merges the two into
    ONE hybrid inverted index whose value lists mix ints and strs.

Comparing that mixed-type set with exact-type membership (``x in set`` or
pandas ``Series.isin``) silently drops matches: ``"351" in {351}`` is
False, and an int ``scene_id`` column never matches a set of str ids.

The fix is one canonical representation applied at the consume boundary.
We choose **string keys** because the JSON/index interop layer (manual
annotation object keys, ``/media`` URLs, template lookups) already speaks
strings; stringifying ints is lossless and total, whereas the reverse
(``int("foo")``) is not.

Design choice: normalize on *consume*, not on *store*. Rewriting how
``build_tag_index`` / annotations persist would risk invalidating existing
generated artefacts. These helpers normalize when the index is read for
filtering, leaving stored files byte-identical.

These are catalog/service utilities. They live in ``src/cinemateca`` (not
``api/``) so the HTTP-agnostic core (``embeddings.SemanticSearch``) can
import them without a layering inversion. A future ``api/services/`` layer
can re-export from here with zero churn.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def scene_id_key(value: Any) -> str:
    """Return the canonical string key for a scene-ID value.

    Accepts Python ``int``/``str``, ``numpy`` integer/float scalars, and
    float-like values. Integral floats have their trailing ``.0`` stripped
    (``351.0 -> "351"``): scene IDs are conceptually integers, and pandas
    yields ``float64`` for an int column that ever held a NaN, so a naive
    ``str(351.0)`` would produce ``"351.0"`` and never match ``"351"``.
    Non-integral floats are left as-is (they should not occur for scene
    IDs, but silently truncating would hide upstream corruption).

    Surrounding whitespace is stripped so a stray ``" 351 "`` from a
    hand-edited annotations file still matches.
    """
    # bool is an int subclass — exclude it explicitly; a bool scene id is
    # always upstream corruption and "True"/"False" keys would be silent.
    if isinstance(value, bool):
        return str(value)

    # int (incl. numpy integer, which is not a Python int but has __index__)
    if isinstance(value, int):
        return str(value)
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return str(int(value))
        if isinstance(value, np.floating):
            value = float(value)
    except ImportError:  # pragma: no cover - numpy is a hard dep here
        pass

    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)

    s = str(value).strip()
    # A stringified integral float ("351.0") from JSON / hand edits.
    if s.endswith(".0") and s[:-2].lstrip("-").isdigit():
        return s[:-2]
    return s


def normalize_tag_index(
    index: Mapping[str, Iterable[Any]] | None,
) -> dict[str, set[str]]:
    """Normalize an inverted tag index to ``{tag: {canonical str id, ...}}``.

    Applied wherever the merged/loaded tag index is consumed for filtering
    (the scenes route and the value passed into
    ``SemanticSearch.combined``) so every membership test is str-vs-str.
    Deduplicates ids that differ only by source type (int ``351`` and str
    ``"351"`` collapse to one ``"351"``).
    """
    if not index:
        return {}
    return {
        str(tag): {scene_id_key(v) for v in ids}
        for tag, ids in index.items()
    }
