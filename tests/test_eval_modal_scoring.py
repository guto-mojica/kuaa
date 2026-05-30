"""Hermetic unit tests for the per-modality scorer logic (E3b).

These cover the pure scoring core of ``cinemateca.eval.retrieval`` — the
``_default_relevance`` resolver, the shared ``_run_modal_eval`` loop, the
``relevance_resolver`` injection hook E2 depends on, the ``film_slug`` scope
filter, and the "no scorable slate" error path — WITHOUT loading any model or
real index. ``generate_slate`` (the only heavy collaborator) is monkeypatched
at the name it is bound to inside ``retrieval`` (the module imports it as a
top-level symbol, so ``cinemateca.eval.retrieval.generate_slate`` is the patch
target). The GPU end-to-end coverage lives in the ``@pytest.mark.acceptance``
``test_eval_multimodal_scoring`` GATE; this file is the CI-without-index half.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import cinemateca.eval.retrieval as retr
from cinemateca.errors import EvalError
from cinemateca.eval.metrics import RetrievalResult
from cinemateca.eval.retrieval import (
    RetrievalRun,
    _default_relevance,
    _run_modal_eval,
)
from cinemateca.eval.slates import ModalQuery

# ─────────────────────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────────────────────


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
        lang="pt",
        relevant_scene_ids=relevant_scene_ids,
        relevance=relevance or {},
        notes=None,
    )


def _row(scene_id: int, *, film_slug: str = "jeca_tatu", score: float = 1.0) -> dict:
    """A minimal candidate-slate row — only the keys the scorer reads."""
    return {
        "scene_id": scene_id,
        "film_slug": film_slug,
        "score": score,
        "keyframe_url": f"/media/library/{film_slug}/frames/scene_{scene_id:04d}.jpg",
    }


def _cfg() -> SimpleNamespace:
    """A throwaway cfg — the scorer never reads it (generate_slate is faked)."""
    return SimpleNamespace()


# ─────────────────────────────────────────────────────────────────────────────
# _default_relevance — method label per query type
# ─────────────────────────────────────────────────────────────────────────────


def test_default_relevance_hypothesis_for_labelled_text_query():
    """A text query carrying relevant_scene_ids + relevance → 'hypothesis'."""
    q = _modal_query(
        query_type="text",
        relevant_scene_ids=(12, 30),
        relevance={"12": 2.0, "30": 1.0},
    )
    ids, rmap, method = _default_relevance(q, [_row(12), _row(30)])
    assert method == "hypothesis"
    assert ids == ("12", "30")
    assert rmap == {"12": 2.0, "30": 1.0}


def test_default_relevance_known_item_for_image_basename_scene_number():
    """An image query whose basename parses (Scene-012) → 'known_item' target 12."""
    q = _modal_query(
        query_type="image",
        text=None,
        image_path=Path("frames/Mazzaropi-Jeca_Tatu-Scene-012-02.jpg"),
    )
    # rows present but should be ignored: the KI target comes from the filename.
    ids, rmap, method = _default_relevance(q, [_row(999)])
    assert method == "known_item"
    assert ids == ("12",)
    assert rmap == {"12": 1.0}


def test_default_relevance_known_item_for_rhyme_anchor():
    """A rhyme query anchor '<slug>/<scene_id>' → 'known_item' target = anchor sid."""
    q = _modal_query(query_type="rhyme", text=None, anchor="jeca_tatu/12")
    ids, rmap, method = _default_relevance(q, [_row(7, film_slug="other_film")])
    assert method == "known_item"
    assert ids == ("12",)
    assert rmap == {"12": 1.0}


def test_default_relevance_pseudo_for_unlabelled_audio_query_uses_top1():
    """An unlabelled audio query → 'pseudo'; target is the top-1 returned scene."""
    q = _modal_query(query_type="audio", text="orchestral strings")
    ids, rmap, method = _default_relevance(q, [_row(7, score=0.9), _row(3, score=0.4)])
    assert method == "pseudo"
    assert ids == ("7",)
    assert rmap == {"7": 1.0}


def test_default_relevance_empty_case_returns_empty_but_keeps_method_label():
    """No labels + no rows → empty ids/map, method still labelled ('pseudo')."""
    q = _modal_query(query_type="audio", text="x")
    ids, rmap, method = _default_relevance(q, [])
    assert ids == ()
    assert rmap == {}
    assert method == "pseudo"


def test_default_relevance_hypothesis_drops_nonpositive_grades(monkeypatch):
    """M3 guard: a labelled query with no positive grade falls back to flat 1.0.

    Otherwise ``ndcg_at_k`` raises a *bare* ValueError that the CLI's
    EvalError/FileNotFoundError handler would not catch.
    """
    q = _modal_query(
        query_type="text",
        relevant_scene_ids=(12, 30),
        relevance={"12": 0.0, "30": -1.0},  # all non-positive
    )
    ids, rmap, method = _default_relevance(q, [_row(12)])
    assert method == "hypothesis"
    assert ids == ("12", "30")
    # Non-positive grades dropped; flat 1.0 fallback keeps the query scorable.
    assert rmap == {"12": 1.0, "30": 1.0}
    # And the resulting map is safe for ndcg_at_k (no bare ValueError).
    from cinemateca.eval.metrics import ndcg_at_k

    assert ndcg_at_k(("12",), rmap, 10) > 0


def test_default_relevance_hypothesis_keeps_only_positive_grades():
    """Mixed grades: positives kept verbatim, non-positives dropped from the map."""
    q = _modal_query(
        query_type="text",
        relevant_scene_ids=(12, 30, 40),
        relevance={"12": 3.0, "30": 0.0, "40": 1.0},
    )
    _ids, rmap, _method = _default_relevance(q, [])
    assert rmap == {"12": 3.0, "40": 1.0}


# ─────────────────────────────────────────────────────────────────────────────
# _run_modal_eval — relevance_resolver hook + per-query method recording (I2)
# ─────────────────────────────────────────────────────────────────────────────


def _patch_slate(monkeypatch, rows_by_qid: dict[str, list[dict]]):
    """Patch generate_slate (as bound in retrieval) to return canned rows per qid."""

    def _fake_generate_slate(*, query, cfg, library_dir, k):
        return list(rows_by_qid.get(query.id, []))

    monkeypatch.setattr(retr, "generate_slate", _fake_generate_slate)


def test_run_modal_eval_invokes_injected_resolver_once_per_query(monkeypatch):
    """The E2 relevance_resolver hook is called once per scored query and its
    3-tuple ``(ids, map, method)`` is used (not _default_relevance)."""
    calls: list[str] = []

    def stub_resolver(query, rows):
        calls.append(query.id)
        # A method label that _default_relevance would never produce, proving
        # the injected resolver — not the default — drove the result.
        sid = retr.scene_id_key(rows[0]["scene_id"])
        return (sid,), {sid: 1.0}, "e2_proxy"

    queries = [
        _modal_query(qid="img-1", query_type="image", text=None, image_path=Path("a.jpg")),
        _modal_query(qid="img-2", query_type="image", text=None, image_path=Path("b.jpg")),
    ]
    _patch_slate(
        monkeypatch,
        {"img-1": [_row(5), _row(9)], "img-2": [_row(7), _row(2)]},
    )

    run = _run_modal_eval(
        _cfg(),
        queries,
        modality="image",
        library_dir=Path("/unused"),
        film_slug=None,
        relevance_resolver=stub_resolver,
    )

    assert isinstance(run, RetrievalRun)
    assert calls == ["img-1", "img-2"]  # exactly once per scored query
    # The injected method is recorded both in the aggregate and per query (I2).
    assert run.context["relevance_method"] == "e2_proxy"
    assert run.context["relevance_methods_by_query"] == {
        "img-1": "e2_proxy",
        "img-2": "e2_proxy",
    }
    # Metrics block carries the four summary keys + query_count.
    for key in ("query_count", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10"):
        assert key in run.metrics
    assert run.metrics["query_count"] == 2
    # img-1 top-1 is the relevant scene → perfect recall/RR for both queries.
    assert run.metrics["recall_at_5"] == pytest.approx(1.0)
    assert run.metrics["mrr"] == pytest.approx(1.0)


def test_run_modal_eval_defaults_to_default_relevance_when_no_resolver(monkeypatch):
    """Without an injected resolver, _default_relevance drives the method label."""
    q = _modal_query(qid="aud-1", query_type="audio", text="strings")
    _patch_slate(monkeypatch, {"aud-1": [_row(7), _row(3)]})

    run = _run_modal_eval(
        _cfg(),
        [q],
        modality="audio",
        library_dir=Path("/unused"),
        film_slug=None,
    )
    assert run.context["relevance_method"] == "pseudo"
    assert run.context["relevance_methods_by_query"] == {"aud-1": "pseudo"}


def test_run_modal_eval_records_mixed_methods_per_query(monkeypatch):
    """A mixed labelled+pseudo run records each query's method separately so a
    consumer can segregate tautological pseudo scores from real ones (I2)."""
    labelled = _modal_query(
        qid="aud-lab",
        query_type="audio",
        text="labelled",
        relevant_scene_ids=(5,),
        relevance={"5": 1.0},
    )
    unlabelled = _modal_query(qid="aud-pseudo", query_type="audio", text="unlabelled")
    _patch_slate(
        monkeypatch,
        {"aud-lab": [_row(5), _row(9)], "aud-pseudo": [_row(7), _row(2)]},
    )

    run = _run_modal_eval(
        _cfg(),
        [labelled, unlabelled],
        modality="audio",
        library_dir=Path("/unused"),
        film_slug=None,
    )
    # Aggregate blends both tiers (kept for back-compat) ...
    assert run.context["relevance_method"] == "hypothesis+pseudo"
    # ... but the per-query map keeps them distinguishable.
    assert run.context["relevance_methods_by_query"] == {
        "aud-lab": "hypothesis",
        "aud-pseudo": "pseudo",
    }
    # Honesty guarantee: a consumer CAN recover which queries are publishable.
    by_q = run.context["relevance_methods_by_query"]
    real = [qid for qid, mth in by_q.items() if mth != "pseudo"]
    assert real == ["aud-lab"]


# ─────────────────────────────────────────────────────────────────────────────
# film_slug scope filter
# ─────────────────────────────────────────────────────────────────────────────


def test_run_modal_eval_scopes_nonrhyme_rows_to_film_slug(monkeypatch):
    """For a non-rhyme modality, rows from other films are excluded when a
    film_slug is given (the slate walks every film; the scorer scopes after)."""
    q = _modal_query(qid="img-1", query_type="image", text=None, image_path=Path("a.jpg"))
    _patch_slate(
        monkeypatch,
        {
            "img-1": [
                _row(5, film_slug="jeca_tatu", score=0.9),
                _row(8, film_slug="other_film", score=0.8),
                _row(11, film_slug="jeca_tatu", score=0.7),
            ]
        },
    )

    captured: dict[str, RetrievalResult] = {}

    def capturing_resolver(query, rows):
        # The resolver receives the ALREADY-scoped rows — assert the cross-film
        # row was dropped before relevance resolution.
        captured["rows"] = rows
        sid = retr.scene_id_key(rows[0]["scene_id"])
        return (sid,), {sid: 1.0}, "ki"

    run = _run_modal_eval(
        _cfg(),
        [q],
        modality="image",
        library_dir=Path("/unused"),
        film_slug="jeca_tatu",
        relevance_resolver=capturing_resolver,
    )
    scoped_sids = [r["scene_id"] for r in captured["rows"]]
    assert scoped_sids == [5, 11]  # other_film/8 excluded
    # And the ranked ids the metric saw are likewise scoped.
    assert run.query_results[0].ranked_scene_ids == ("5", "11")


def test_run_modal_eval_does_not_scope_rhyme_rows(monkeypatch):
    """Rhyme is cross-film by design — rows from OTHER films are NOT filtered
    even when a film_slug is supplied."""
    q = _modal_query(qid="rh-1", query_type="rhyme", text=None, anchor="jeca_tatu/12")
    _patch_slate(
        monkeypatch,
        {
            "rh-1": [
                _row(3, film_slug="other_film_a", score=0.9),
                _row(4, film_slug="other_film_b", score=0.8),
            ]
        },
    )

    run = _run_modal_eval(
        _cfg(),
        [q],
        modality="rhyme",
        library_dir=Path("/unused"),
        film_slug="jeca_tatu",  # supplied, but rhyme must ignore it
    )
    # Both cross-film rows survive (no scoping) → they are the ranked candidates.
    assert run.query_results[0].ranked_scene_ids == ("3", "4")
    # Default rhyme KI target is the anchor scene (12), which is structurally
    # absent from the cross-film pool → recall/RR are 0 (GATE-completeness only).
    assert run.metrics["recall_at_5"] == pytest.approx(0.0)
    assert run.metrics["mrr"] == pytest.approx(0.0)
    assert run.context["relevance_methods_by_query"] == {"rh-1": "known_item"}


# ─────────────────────────────────────────────────────────────────────────────
# "no scorable slate" path → clean EvalError (not a bare ValueError/traceback)
# ─────────────────────────────────────────────────────────────────────────────


def test_run_modal_eval_raises_evalerror_when_no_query_scorable(monkeypatch):
    """Every query yields empty rows + no labels → clean EvalError, not the bare
    ValueError summarize_results raises on an empty list."""
    # Unlabelled audio queries with empty slates: _default_relevance returns no
    # ids (the empty-case), every query is skipped, results stays empty.
    queries = [
        _modal_query(qid="aud-1", query_type="audio", text="x"),
        _modal_query(qid="aud-2", query_type="audio", text="y"),
    ]
    _patch_slate(monkeypatch, {})  # generate_slate returns [] for every qid

    with pytest.raises(EvalError) as excinfo:
        _run_modal_eval(
            _cfg(),
            queries,
            modality="audio",
            library_dir=Path("/unused"),
            film_slug=None,
        )
    # The message names the modality + how many candidates were checked.
    assert "audio" in str(excinfo.value)
    assert "scorable slate" in str(excinfo.value)


def test_run_modal_eval_rejects_nonpositive_top_k():
    """top_k < 1 is a clean EvalError before any slate work."""
    with pytest.raises(EvalError):
        _run_modal_eval(
            _cfg(),
            [_modal_query()],
            modality="image",
            library_dir=Path("/unused"),
            film_slug=None,
            top_k=0,
        )
