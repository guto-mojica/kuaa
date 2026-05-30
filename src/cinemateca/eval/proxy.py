"""Proxy-relevance labeller for the WS-4 ablation harness (E2a).

The launch ablation table (E2b) must be producible with **zero human grades**,
yet every published number has to be honest about *what* it measures. This module
attaches a relevance label to one :class:`~cinemateca.eval.slates.ModalQuery` from
one of three proxy signals, returning the signal's name alongside the labels so a
consumer (E2b's runner) can segregate rows by honesty tier and never blend them
into a single misleading average.

The three proxy signals (spec §6 E2)
-------------------------------------
**KI — Known-Item.** For a query with an unambiguous single correct scene. The
correct item is the *anchor scene the query came from*: for an ``image`` query
that is the scene whose keyframe is the query image (parsed from the keyframe
basename, e.g. ``...-Scene-089-02.jpg`` -> 89); for a ``rhyme`` query it is the
anchor scene in ``query.anchor`` (``"<slug>/<scene_id>"``).

  .. warning::
     The **rhyme** KI target is *structurally weak*. ``find_rhymes(
     cross_film_only=True)`` excludes the anchor film, so the anchor scene can
     **never** appear in a cross-film rhyme slate — recall and reciprocal-rank
     are 0 by construction, regardless of retriever quality. KI is a meaningful
     signal for ``image`` queries; for ``rhyme`` it certifies only that the
     modality runs end-to-end. E2b will NOT publish a rhyme KI row.

**PR — Pseudo-Relevance.** The top-1 of a **reference retriever** (supplied by the
caller via ``reference_hits``) is treated as relevant, to measure whether *other*
retriever variants AGREE with the reference. This is a RELATIVE-agreement proxy,
used only for retriever-variant comparison rows.

  .. warning::
     PR locks onto the **reference** retriever's top-1, NOT the query's own
     top-1. A different retriever that ranks the reference's top-1 below #1 is a
     real, non-tautological miss. (An earlier draft —
     ``cinemateca.eval.retrieval._default_relevance`` — used the scored
     retriever's *own* top-1 as the label, which is tautological: recall@k and
     RR are 1.0 by construction. PR here MUST NOT do that; the label comes from
     ``reference_hits``, an argument independent of the row being scored.)

**HY — Hypothesis.** The maintainer's ``relevance`` / ``relevant_scene_ids`` from
``m3_*_queries.yaml``, used as-is for text rows. These are *pre-curator
hypotheses* — best-guess relevant scenes recorded before any grading session —
and the run honesty tier must label them as such, never as ground truth.

Layering: core (``cinemateca.*``); MUST NOT import ``api.*`` (import-linter).
Hermetic: attaching a label never loads a model or reads the retrieval index —
the labeller only canonicalises ids and parses basenames.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from cinemateca.eval.slates import ModalQuery
from cinemateca.scene_ids import scene_id_key

# Scene-number extractor for an image keyframe basename (e.g.
# "...Scene-089-02.jpg" -> 89). Inlined here rather than imported from the
# sibling ``cinemateca.rhymes.algorithm`` (whose copy is underscore-private):
# the project invariant keeps a package's public surface in its __init__, and
# .importlinter prefers inlining a one-line helper over a cross-package
# carve-out. Mirrors ``cinemateca.rhymes.algorithm._SCENE_NUM_RE`` and the same
# inline in ``cinemateca.eval.retrieval``.
_SCENE_NUM_RE = re.compile(r"[Ss]cene[-_](\d+)")


def _positive_relevance(relevance: dict[Any, Any]) -> dict[str, float]:
    """Canonicalise a relevance map, dropping non-positive grades.

    ``ndcg_at_k`` raises a bare ``ValueError`` when handed an all-non-positive
    map (it needs at least one grade > 0 to form an ideal ranking), so every
    proxy signal funnels its relevance through here before returning. Keys are
    canonicalised via :func:`cinemateca.scene_ids.scene_id_key`; a grade that is
    0 or negative is dropped entirely.
    """
    out: dict[str, float] = {}
    for key, value in relevance.items():
        grade = float(value)
        if grade > 0:
            out[scene_id_key(key)] = grade
    return out


def _hypothesis(query: ModalQuery) -> tuple[tuple[str, ...], dict[str, float]]:
    """HY signal: the maintainer's YAML ``relevant_scene_ids`` + ``relevance``.

    Ids are canonicalised; the relevance map keeps only positive grades. When
    the maintainer recorded ``relevant_scene_ids`` but no usable positive
    ``relevance`` map (all-zero / missing), fall back to a flat ``1.0`` per
    relevant id so the labelled query stays scorable (binary relevance).
    """
    rel_ids = tuple(scene_id_key(s) for s in query.relevant_scene_ids)
    relevance = _positive_relevance(query.relevance)
    if not relevance and rel_ids:
        relevance = {sid: 1.0 for sid in rel_ids}
    return rel_ids, relevance


def _known_item(query: ModalQuery) -> tuple[tuple[str, ...], dict[str, float]]:
    """KI signal: the single anchor scene the query came from.

    * **image** — the scene number parsed from the ``image_path`` basename
      (``...-Scene-089-02.jpg`` -> 89) via :data:`_SCENE_NUM_RE`.
    * **rhyme** — the anchor scene id in ``query.anchor`` (``"<slug>/<scene_id>"``).
      See the module docstring: this target is structurally unreachable in a
      cross-film rhyme slate, so a rhyme KI row scores 0 by construction.

    Returns empty labels (``(), {}``) when no anchor scene can be derived (an
    image basename that does not match, or a malformed/absent anchor) so the
    caller can skip the query rather than fabricate a label.
    """
    if query.query_type == "image" and query.image_path is not None:
        match = _SCENE_NUM_RE.search(query.image_path.name)
        if match:
            sid = scene_id_key(int(match.group(1)))
            return (sid,), {sid: 1.0}
    elif query.query_type == "rhyme" and query.anchor and query.anchor.count("/") == 1:
        _slug, sid_s = query.anchor.split("/", 1)
        try:
            sid = scene_id_key(int(sid_s))
        except ValueError:
            return (), {}
        return (sid,), {sid: 1.0}
    return (), {}


def _pseudo_relevance(
    reference_hits: list[dict] | None,
) -> tuple[tuple[str, ...], dict[str, float]]:
    """PR signal: the REFERENCE retriever's top-1 is treated as relevant.

    ``reference_hits`` is the ranked output of a *reference* retriever (each row
    a dict with a ``"scene_id"`` key, matching ``search_audio`` /
    ``generate_slate`` output); the top-1 is ``reference_hits[0]``. This is a
    relative-agreement label — independent of whatever retriever is being scored
    against it — NOT the scored retriever's own top-1 (that would be tautological;
    see the module docstring). Returns empty labels when the reference produced
    nothing.
    """
    if not reference_hits:
        return (), {}
    sid = scene_id_key(reference_hits[0]["scene_id"])
    return (sid,), {sid: 1.0}


def proxy_labels(
    query: ModalQuery,
    *,
    library_dir: Path,
    cfg: Any,
    reference_hits: list[dict] | None = None,
) -> tuple[tuple[str, ...], dict[str, float], str]:
    """Return ``(relevant_scene_ids, relevance, proxy_method)`` for one query.

    ``proxy_method`` is one of ``"KI"`` / ``"PR"`` / ``"HY"`` and records which
    proxy signal produced the labels, so E2b can segregate rows by honesty tier.
    ``relevant_scene_ids`` are canonical string ids (:func:`scene_id_key`);
    ``relevance`` maps canonical id -> POSITIVE float grade only (non-positive
    grades dropped to keep ``ndcg_at_k`` from raising).

    Dispatch (first match wins):

      1. query carries HY labels (``relevance`` or ``relevant_scene_ids``
         non-empty) -> **HY** (maintainer's pre-curator hypothesis).
      2. ``query_type`` in ``{"image", "rhyme"}`` -> **KI** (the anchor scene).
      3. ``reference_hits`` supplied (audio / fusion, or any variant-comparison
         row) -> **PR** (the reference retriever's top-1).
      4. otherwise -> ``((), {}, <method>)`` with empty labels; the caller
         skips or handles the unlabelled query.

    The HY-before-KI order means a labelled query is always HY regardless of its
    type; the KI-before-PR order means an image/rhyme query keeps its honest
    known-item anchor even when a reference hit list is also supplied (PR is the
    fallback for modalities that have no known-item anchor). ``library_dir`` and
    ``cfg`` are accepted for signature parity with the slate/scorer layer and a
    future signal that needs them; the current three signals are hermetic and
    read neither.
    """
    if query.relevance or query.relevant_scene_ids:
        rel_ids, relevance = _hypothesis(query)
        return rel_ids, relevance, "HY"

    if query.query_type in ("image", "rhyme"):
        rel_ids, relevance = _known_item(query)
        return rel_ids, relevance, "KI"

    if reference_hits:
        rel_ids, relevance = _pseudo_relevance(reference_hits)
        return rel_ids, relevance, "PR"

    return (), {}, "PR"


__all__ = ["proxy_labels"]
