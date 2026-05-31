"""Hermetic unit tests for the proxy-relevance labeller (E2a).

These cover ``cinemateca.eval.proxy.proxy_labels`` — the per-query labeller that
supplies ``(relevant_scene_ids, relevance, proxy_method)`` to the eval scorers so
the WS-4 ablation table is producible with ZERO human grades, each row tagged with
which proxy signal produced it (KI / PR / HY).

No model is loaded: KI parses a keyframe basename, PR reads a supplied reference
hit list, HY passes the maintainer's YAML hypotheses through. ``cfg`` is an inert
``SimpleNamespace`` because the labeller never touches the retrieval backend — it
only attaches labels.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from cinemateca.eval import proxy_labels
from cinemateca.eval.slates import ModalQuery
from cinemateca.scene_ids import scene_id_key

# Real keyframe basename for the m3_full image-09 entry (Jeca scene 89). The
# anchor scene is encoded in the basename — KI must parse 89 out of it.
_IMAGE_09_PATH = Path(
    "data/library/jeca_tatu/frames/scenes/keyframes_content/"
    "Mazzaropi-Jeca_Tatu_Paixão_Flix-Scene-089-02.jpg"
)


def _modal_query(
    *,
    qid: str = "q",
    query_type: str = "text",
    text: str | None = "a query",
    image_path: Path | None = None,
    anchor: str | None = None,
    w: float | None = None,
    relevant_scene_ids: tuple[int, ...] = (),
    relevance: dict[str, float] | None = None,
) -> ModalQuery:
    return ModalQuery(
        id=qid,
        query_type=query_type,
        text=text,
        image_path=image_path,
        anchor=anchor,
        w=w,
        lang="en",
        relevant_scene_ids=relevant_scene_ids,
        relevance=relevance or {},
        notes=None,
    )


def test_known_item_image_query_targets_anchor_scene() -> None:
    """An image query's KI target is the anchor scene parsed from its basename.

    Hermetic: no model load. KI just parses ``Scene-089`` -> 89 out of the
    real m3_full image-09 keyframe basename.
    """
    q = _modal_query(qid="image-09", query_type="image", image_path=_IMAGE_09_PATH)

    rel_ids, relevance, method = proxy_labels(
        q, library_dir=Path("data/library"), cfg=SimpleNamespace()
    )

    assert method == "KI"
    assert scene_id_key(89) in rel_ids
    assert relevance[scene_id_key(89)] > 0


def test_hypothesis_label_passthrough_for_text() -> None:
    """A text query carrying YAML hypotheses is labelled HY, verbatim+canonical."""
    q = _modal_query(
        qid="text-01",
        query_type="text",
        relevant_scene_ids=(34,),
        relevance={"34": 2.0},
    )

    rel_ids, relevance, method = proxy_labels(
        q, library_dir=Path("data/library"), cfg=SimpleNamespace()
    )

    assert method == "HY"
    assert rel_ids == (scene_id_key(34),)
    assert relevance == {scene_id_key(34): 2.0}


def test_pseudo_relevance_uses_reference_top1() -> None:
    """PR treats the REFERENCE retriever's top-1 as relevant — not self-top-1.

    The reference hit list is supplied independently of any retriever the row
    is comparing; PR must lock onto ``reference_hits[0]`` (scene 7 here), so a
    variant that ranks scene 7 below #1 produces a real, non-tautological miss.

    An unlabelled ``text`` query (not image/rhyme, so no known-item anchor) is
    the PR fallback path.
    """
    q = _modal_query(qid="text-pr", query_type="text", text="rural ambience")
    reference_hits = [{"scene_id": 7, "score": 0.9}, {"scene_id": 3, "score": 0.5}]

    rel_ids, relevance, method = proxy_labels(
        q,
        library_dir=Path("data/library"),
        cfg=SimpleNamespace(),
        reference_hits=reference_hits,
    )

    assert method == "PR"
    assert rel_ids == (scene_id_key(7),)
    assert relevance == {scene_id_key(7): 1.0}


def test_hypothesis_drops_non_positive_grades() -> None:
    """HY drops non-positive grades so ``ndcg_at_k`` never sees an all-zero map."""
    q = _modal_query(
        qid="text-02",
        query_type="text",
        relevant_scene_ids=(5, 9),
        relevance={"5": 0.0, "9": 2.0},
    )

    rel_ids, relevance, method = proxy_labels(
        q, library_dir=Path("data/library"), cfg=SimpleNamespace()
    )

    assert method == "HY"
    assert relevance == {scene_id_key(9): 2.0}
    assert scene_id_key(5) not in relevance
