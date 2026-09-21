"""Phase 0 — the pool must be worth grading before anyone grades it.

Grading is the expensive, non-repeatable input: a curator's attention spent on
a corrupt pool cannot be spent again. Everything pinned here either corrupts
judgments already collected or wastes the session that collects them.

Covers:

* **0.1** scene identity — a cut edit renumbers scenes, so grades must be
  relabelled and a pool whose numbering moved must refuse the join.
* **0.3** blinded order stable under pool edits.
* **0.5** cross-film merge is rank-based, and its interleave is specified.
* **0.7** ``_cfg_with`` validates override keys; the composition report fails
  loudly on a variant that widens the pool by nothing.
* grade keys are film-qualified — a pool spans films and ``scene_id`` repeats.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kuaa.errors import EvalError
from kuaa.eval.composition import analyse_pool
from kuaa.eval.grades import EvalRun, Grade, grade_scene_key, load_run, save_grade
from kuaa.eval.registry import POOL_VARIANTS, RETRIEVER_REGISTRY, RetrieverVariant
from kuaa.eval.scene_manifest import (
    SceneManifestMismatch,
    build_manifest,
    require_manifest_match,
    scene_manifest_hash,
)
from kuaa.eval.slates import _cfg_with, _merge_across_films, blind

# ── helpers ──────────────────────────────────────────────────────────────────


def _row(scene_id: int, *, film_slug: str = "film_a", score: float = 1.0) -> dict:
    return {"scene_id": scene_id, "film_slug": film_slug, "score": score}


def _write_keyframes(library_dir: Path, slug: str, boundaries: dict[int, tuple[int, int]]) -> None:
    """Write a minimal ``keyframes_metadata.json`` with the given scene bounds."""
    meta_dir = library_dir / slug / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "scene_id": sid,
            "keyframe_id": f"scene_{sid:04d}_kf_01",
            "filepath": str(library_dir / slug / "frames" / f"{sid}.jpg"),
            "start_frame": start,
            "end_frame": end,
        }
        for sid, (start, end) in sorted(boundaries.items())
    ]
    (meta_dir / "keyframes_metadata.json").write_text(json.dumps(rows), encoding="utf-8")


# ── 0.3 — blinded order is stable under a pool edit ───────────────────────────


def test_blinding_order_survives_adding_one_candidate():
    """Adding a candidate must move only that candidate, not repermute the queue.

    ``blind`` used to shuffle the *list*, so one extra row repermuted
    everything — breaking the shuffle's own promise that a resuming grader
    meets the same queue, on exactly the edit that promise exists for. It is
    also what makes re-grading a changed variant affordable: only the diff
    needs a second pass.
    """
    base = [_row(i) for i in range(1, 13)]
    before = [r["scene_id"] for r in blind(base, seed="run-1:q1")]
    after = [r["scene_id"] for r in blind([*base, _row(99)], seed="run-1:q1")]

    assert 99 in after
    assert [s for s in after if s != 99] == before, (
        "the pre-existing queue order must be untouched by an insertion"
    )


def test_blinding_order_survives_removing_one_candidate():
    """The same guarantee in the other direction."""
    base = [_row(i) for i in range(1, 13)]
    before = [r["scene_id"] for r in blind(base, seed="run-1:q1")]
    after = [r["scene_id"] for r in blind([r for r in base if r["scene_id"] != 5], seed="run-1:q1")]
    assert after == [s for s in before if s != 5]


def test_blinding_order_is_per_candidate_not_per_position():
    """Two films' scene 5 are different candidates and must order independently."""
    rows = [_row(5, film_slug="film_a"), _row(5, film_slug="film_b")]
    out = blind(rows, seed="s")
    assert {(r["film_slug"], r["scene_id"]) for r in out} == {("film_a", 5), ("film_b", 5)}


# ── 0.5 — cross-film merge is rank-based, interleave is specified ─────────────


def test_cross_film_merge_is_round_robin_by_sorted_slug():
    """Rank 1 from every film, then rank 2, films in sorted-slug order.

    Sorting a flat cross-film list by raw ``score`` is only defensible for
    CLIP: BM25 carries per-film IDF and a fused RRF score is a per-film rank
    transform, so neither is comparable between films. Here ``zeta``'s scores
    dwarf ``alpha``'s; a score sort would return zeta's whole list first.
    """
    per_film = {
        "zeta": [_row(1, film_slug="zeta", score=99.0), _row(2, film_slug="zeta", score=98.0)],
        "alpha": [_row(1, film_slug="alpha", score=0.01), _row(2, film_slug="alpha", score=0.001)],
    }
    merged = _merge_across_films(per_film, k=4)
    assert [(r["film_slug"], r["scene_id"]) for r in merged] == [
        ("alpha", 1),
        ("zeta", 1),
        ("alpha", 2),
        ("zeta", 2),
    ]


def test_cross_film_merge_truncates_after_interleaving():
    """The ``k`` cut lands on the interleaved order, so every film reaches it."""
    per_film = {
        "zeta": [_row(i, film_slug="zeta", score=99.0 - i) for i in range(1, 6)],
        "alpha": [_row(i, film_slug="alpha", score=0.01) for i in range(1, 6)],
    }
    merged = _merge_across_films(per_film, k=2)
    assert {r["film_slug"] for r in merged} == {"alpha", "zeta"}, (
        "a score sort would have filled k entirely from the high-scoring film"
    )


def test_pool_candidates_output_order_is_the_fused_order(monkeypatch, tmp_path):
    """Dict insertion order is load-bearing: it is the order ``blind`` is handed."""
    import kuaa.eval.slates as slates
    from kuaa.eval.slates import pool_candidates
    from kuaa.search import Query

    class _Hit:
        def __init__(self, sid, score):
            self.scene_id, self.score = sid, score
            self.film_slug = "film_a"

    per_variant = {"clip": [(7, 0.9), (3, 0.8)], "bm25": [(3, 0.7), (5, 0.6)]}

    def _fake_aggregate(query, *, cfg, mode, top_k, weights=None, **kw):
        return SimpleNamespace(hits=[_Hit(s, sc) for s, sc in per_variant.get(mode, [])])

    monkeypatch.setattr(slates, "aggregate", _fake_aggregate)

    rows = pool_candidates(
        q=Query.of_text("x"),
        cfg=SimpleNamespace(),
        library_dir=tmp_path,
        k=9,
        load_meta=lambda slug: slates._empty_meta(slug, tmp_path),
        variants=(RETRIEVER_REGISTRY["clip"], RETRIEVER_REGISTRY["bm25"]),
    )
    # clip's ranking first (7, 3), then bm25's candidates clip did not
    # propose (5). Scene 3 keeps clip's position — first proposer wins.
    assert [r["scene_id"] for r in rows] == [7, 3, 5]


# ── 0.7 — a variant whose config is inert must fail loudly ────────────────────


class _SearchCfg:
    """Stand-in for the pydantic ``SearchCfg`` — declares its fields via __dict__."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_cfg_with_rejects_an_override_key_the_config_does_not_have():
    """``model_copy(update=...)`` does not validate, so a typo'd key was set
    silently and the variant quietly became a duplicate of plain ``hybrid``."""
    cfg = SimpleNamespace(search=_SearchCfg(hybrid_metadata_w=0.65))
    with pytest.raises(EvalError, match="hybrid_metdata_w"):
        _cfg_with(cfg, hybrid_metdata_w=0.0)  # codespell:ignore


def test_cfg_with_accepts_a_key_the_config_declares():
    cfg = SimpleNamespace(search=_SearchCfg(hybrid_metadata_w=0.65))
    out = _cfg_with(cfg, hybrid_metadata_w=0.0)
    assert out.search.hybrid_metadata_w == 0.0
    assert cfg.search.hybrid_metadata_w == 0.65, "the source config must not be mutated"


def test_real_settings_reject_a_typoed_override():
    """The same guard on the shipped pydantic model, not just the test double."""
    from kuaa.config import load_config

    cfg = load_config(Path("config/default.yaml"), project_root=Path.cwd(), ensure_dirs=False)
    with pytest.raises(EvalError):
        _cfg_with(cfg, hybrid_metadata_weight=0.0)


def _pool_record(qid: str, rows: list[dict]) -> dict:
    return {"id": qid, "query_type": "text", "results": rows}


def test_composition_report_fails_a_variant_that_widens_the_pool_by_nothing():
    records = [
        _pool_record(
            "q1",
            [
                {"scene_id": 1, "film_slug": "f", "pool": {"clip": 1, "bm25": 1}},
                {"scene_id": 2, "film_slug": "f", "pool": {"clip": 2, "bm25": 2}},
            ],
        )
    ]
    variants = (RETRIEVER_REGISTRY["clip"], RETRIEVER_REGISTRY["bm25"])
    report = analyse_pool(records, run="r", variants=variants)
    assert not report.ok
    assert report.identical_pairs == [("clip", "bm25")]
    assert any("identical rank maps" in f for f in report.failures)


def test_composition_report_passes_a_pool_whose_variants_differ():
    records = [
        _pool_record(
            "q1",
            [
                {"scene_id": 1, "film_slug": "f", "pool": {"clip": 1}},
                {"scene_id": 2, "film_slug": "f", "pool": {"bm25": 1}},
            ],
        )
    ]
    variants = (RETRIEVER_REGISTRY["clip"], RETRIEVER_REGISTRY["bm25"])
    report = analyse_pool(records, run="r", variants=variants)
    assert report.ok, report.failures


def test_composition_report_does_not_fault_a_derived_variant_for_adding_nothing():
    """A fusion or a reranker cannot extend the candidate set it works over.

    Zero unique candidates is *correct* behaviour for both, so the
    unique-contribution check applies only to source retrievers. It is not a
    hypothetical exemption: on the shipped corpus01 pool ``hybrid`` proposes
    560 candidates and 0 unique ones. Distinguishability still applies, and
    here the reranker earns it by reordering.
    """
    records = [
        _pool_record(
            "q1",
            [
                {"scene_id": 1, "film_slug": "f", "pool": {"hybrid": 1, "hybrid_rerank": 2}},
                {"scene_id": 2, "film_slug": "f", "pool": {"hybrid": 2, "hybrid_rerank": 1}},
            ],
        )
    ]
    variants = (RETRIEVER_REGISTRY["hybrid"], RETRIEVER_REGISTRY["hybrid_rerank"])
    report = analyse_pool(records, run="r", variants=variants)
    assert report.ok, report.failures
    by_name = {v.name: v for v in report.variants}
    assert by_name["hybrid_rerank"].unique_candidates == 0
    assert by_name["hybrid"].unique_candidates == 0


def test_composition_report_still_faults_a_source_that_adds_nothing():
    """The exemption is for derived variants only — a source must widen the pool."""
    records = [
        _pool_record(
            "q1",
            [{"scene_id": 1, "film_slug": "f", "pool": {"clip": 1, "bm25": 1}}],
        ),
        _pool_record(
            "q2",
            [{"scene_id": 2, "film_slug": "f", "pool": {"clip": 1, "bm25": 2}}],
        ),
    ]
    variants = (RETRIEVER_REGISTRY["clip"], RETRIEVER_REGISTRY["bm25"])
    report = analyse_pool(records, run="r", variants=variants)
    assert not report.ok
    assert any("source retriever" in f for f in report.failures)


def test_composition_report_catches_an_inert_reranker():
    """Same candidates AND same ranks on every query — the shipped defect."""
    records = [
        _pool_record(
            f"q{i}",
            [{"scene_id": 1, "film_slug": "f", "pool": {"hybrid": 1, "hybrid_rerank": 1}}],
        )
        for i in range(3)
    ]
    variants = (RETRIEVER_REGISTRY["hybrid"], RETRIEVER_REGISTRY["hybrid_rerank"])
    report = analyse_pool(records, run="r", variants=variants)
    assert not report.ok
    assert ("hybrid", "hybrid_rerank") in report.identical_pairs


def test_composition_report_skips_the_audit_for_a_single_retriever_modality():
    """Image and rhyme runs have no span to audit — an empty report, not a failure."""
    report = analyse_pool(
        [{"id": "i1", "query_type": "image", "results": []}], run="r", expect_text=False
    )
    assert report.ok


def test_registry_is_the_only_variant_name_space():
    assert [v.name for v in POOL_VARIANTS] == list(RETRIEVER_REGISTRY)
    assert all(isinstance(v, RetrieverVariant) for v in POOL_VARIANTS)


# ── grade keys must be film-qualified ────────────────────────────────────────


def test_grade_key_is_film_qualified():
    """A pool spans the library, so ``scene_id`` alone is not a key.

    The shipped ``corpus01`` pool has 104 places where two films' scenes share
    a scene_id inside one query. With a bare-id key, grading one of them
    records a judgment about the other.
    """
    assert grade_scene_key("coisas_nossas_1931", 45) == "coisas_nossas_1931/45"
    assert grade_scene_key("chronopolis_1982", 45) != grade_scene_key("coisas_nossas_1931", 45)


def test_grade_key_falls_back_to_the_bare_id_without_a_slug():
    assert grade_scene_key("", 45) == "45"
    assert grade_scene_key(None, 45) == "45"


def test_two_films_same_scene_id_are_two_grades(tmp_path):
    run = EvalRun(run_id="r", root=tmp_path)
    save_grade(
        run,
        query_id="q1",
        scene_id=grade_scene_key("film_a", 45),
        grader="rg",
        grade=Grade.HIGHLY_RELEVANT,
    )
    save_grade(
        run,
        query_id="q1",
        scene_id=grade_scene_key("film_b", 45),
        grader="rg",
        grade=Grade.IRRELEVANT,
    )
    loaded = load_run(run)
    assert loaded.grades[("q1", "film_a/45")].grade == Grade.HIGHLY_RELEVANT
    assert loaded.grades[("q1", "film_b/45")].grade == Grade.IRRELEVANT


# ── 0.1 — scene identity survives (or loudly refuses) a cut edit ──────────────


def test_scene_manifest_hash_changes_when_a_boundary_moves(tmp_path):
    _write_keyframes(tmp_path, "f", {1: (0, 100), 2: (100, 200)})
    before = scene_manifest_hash(tmp_path, "f")
    _write_keyframes(tmp_path, "f", {1: (0, 150), 2: (150, 200)})
    after = scene_manifest_hash(tmp_path, "f")
    assert before and after and before != after


def test_scene_manifest_hash_is_stable_across_rereads(tmp_path):
    _write_keyframes(tmp_path, "f", {1: (0, 100), 2: (100, 200)})
    assert scene_manifest_hash(tmp_path, "f") == scene_manifest_hash(tmp_path, "f")


def test_scene_manifest_hash_is_none_for_a_film_without_metadata(tmp_path):
    assert scene_manifest_hash(tmp_path, "missing") is None


def test_manifest_match_refuses_a_pool_whose_numbering_moved(tmp_path):
    _write_keyframes(tmp_path, "f", {1: (0, 100), 2: (100, 200)})
    stored = build_manifest(tmp_path, ["f"])
    require_manifest_match(stored, tmp_path, run="r")  # unchanged → passes

    # A merge: two scenes become one, and every scene after it renumbers.
    _write_keyframes(tmp_path, "f", {1: (0, 200)})
    with pytest.raises(SceneManifestMismatch, match="scene numbering changed"):
        require_manifest_match(stored, tmp_path, run="r")


def test_manifest_match_passes_an_unpinned_pool_with_a_warning(tmp_path, caplog):
    """A pool written before the guard cannot be retroactively pinned.

    Refusing to grade every such pool would be worse than saying so.
    """
    with caplog.at_level("WARNING"):
        require_manifest_match({}, tmp_path, run="legacy")
    assert "not pinned" in caplog.text


def test_cut_edit_relabels_grades_through_the_old_to_new_map(tmp_path):
    """Grades are the third per-scene artifact keyed by the scene ordinal.

    Scene 5 shifts to 4 (a merge upstream deleted a boundary) and scene 9 is
    orphaned. The relabelled grade wins the last-write-wins reduce at its new
    ordinal; the vacated ordinals no longer carry a judgment about whatever
    scene now holds that number.
    """
    from kuaa.preprocess.service import _migrate_eval_grades

    root = tmp_path / "eval"
    run = EvalRun(run_id="corpus01", root=root)
    for sid, grade in ((5, Grade.HIGHLY_RELEVANT), (9, Grade.RELEVANT)):
        save_grade(
            run,
            query_id="q1",
            scene_id=grade_scene_key("jeca", sid),
            grader="rg",
            grade=grade,
        )

    ctx = SimpleNamespace(slug="jeca")
    cfg = SimpleNamespace(eval=SimpleNamespace(root=str(root)))
    _migrate_eval_grades(ctx, cfg, {5: 4})

    grades = load_run(run).grades
    assert grades[("q1", "jeca/4")].grade == Grade.HIGHLY_RELEVANT, "shifted scene relabelled"
    assert grades[("q1", "jeca/5")].grade == Grade.SKIP, "the vacated ordinal is neutralised"
    assert grades[("q1", "jeca/9")].grade == Grade.SKIP, "the orphaned scene is dropped"
    assert grades[("q1", "jeca/4")].grader == "rg", "grader identity survives, so IAA still works"


def test_cut_edit_neutralises_before_it_relabels(tmp_path):
    """Order between the two write passes is the correctness argument.

    Scene 11 is orphaned and scene 10 shifts onto ordinal 11. Interleaved by
    scene id, the SKIP for 11 would land on top of the grade that just moved
    in and destroy it.
    """
    from kuaa.preprocess.service import _migrate_eval_grades

    root = tmp_path / "eval"
    run = EvalRun(run_id="c", root=root)
    save_grade(run, query_id="q", scene_id="jeca/10", grader="rg", grade=Grade.HIGHLY_RELEVANT)
    save_grade(run, query_id="q", scene_id="jeca/11", grader="rg", grade=Grade.WEAKLY)

    _migrate_eval_grades(
        SimpleNamespace(slug="jeca"),
        SimpleNamespace(eval=SimpleNamespace(root=str(root))),
        {10: 11},
    )
    grades = load_run(run).grades
    assert grades[("q", "jeca/11")].grade == Grade.HIGHLY_RELEVANT
    assert grades[("q", "jeca/10")].grade == Grade.SKIP


def test_cut_edit_leaves_another_films_grades_alone(tmp_path):
    from kuaa.preprocess.service import _migrate_eval_grades

    root = tmp_path / "eval"
    run = EvalRun(run_id="c", root=root)
    save_grade(run, query_id="q", scene_id="other/5", grader="rg", grade=Grade.RELEVANT)

    _migrate_eval_grades(
        SimpleNamespace(slug="jeca"),
        SimpleNamespace(eval=SimpleNamespace(root=str(root))),
        {5: 4},
    )
    grades = load_run(run).grades
    assert grades[("q", "other/5")].grade == Grade.RELEVANT
    assert ("q", "other/4") not in grades


def test_composition_report_fails_when_a_film_never_reaches_the_pool():
    """``k`` below the film count excludes films alphabetically, not by relevance.

    The cross-film merge interleaves round-robin by sorted slug, so the ``k``
    cut lands mid-rotation: on an 11-film library at ``k=9`` the two
    last-sorting films are absent from *every* query in the run. No
    per-variant check can see that — it is a property of ``k`` and the corpus
    — so it has to be its own check rather than the operator's memory of what
    ``--k`` should have been.
    """
    records = [
        _pool_record("q1", [{"scene_id": 1, "film_slug": "aaa", "pool": {"clip": 1}}]),
        _pool_record("q2", [{"scene_id": 2, "film_slug": "aaa", "pool": {"bm25": 1}}]),
    ]
    variants = (RETRIEVER_REGISTRY["clip"], RETRIEVER_REGISTRY["bm25"])
    report = analyse_pool(
        records, run="r", variants=variants, library_films=["aaa", "zzz_last_alphabetically"]
    )
    assert not report.ok
    assert report.missing_films == ["zzz_last_alphabetically"]
    assert any("raise --k" in f for f in report.failures)


def test_composition_report_passes_when_every_searched_film_appears():
    records = [
        _pool_record(
            "q1",
            [
                {"scene_id": 1, "film_slug": "aaa", "pool": {"clip": 1}},
                {"scene_id": 2, "film_slug": "zzz", "pool": {"bm25": 1}},
            ],
        )
    ]
    variants = (RETRIEVER_REGISTRY["clip"], RETRIEVER_REGISTRY["bm25"])
    report = analyse_pool(records, run="r", variants=variants, library_films=["aaa", "zzz"])
    assert report.ok, report.failures
    assert report.missing_films == []
