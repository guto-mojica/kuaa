"""Hermetic tests for the per-modality slate-generation layer (E3a).

Two concerns are covered:

1. ``load_modal_queries`` — parses an ``m3_full``-shaped YAML (top-level
   ``queries:`` list) into frozen :class:`ModalQuery` objects, validating
   per query type and raising :class:`EvalError` on malformed entries.
   The fixtures copy one representative entry per type from the real
   ``data/eval/m3_full_queries.yaml`` so the parser is tested against the
   actual on-disk shape, not an invented one.

2. ``generate_slate`` — dispatches on ``query.query_type`` to the real
   retrieval backend for that modality and maps the results into the
   9-key rows-template contract the ``/eval`` UI renders. These tests are
   hermetic: the backends (``find`` / ``find_rhymes``) are monkeypatched
   with fakes, and the cfg is a ``SimpleNamespace``. No real models or
   indexes are loaded.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cinemateca.errors import EvalError
from cinemateca.eval.slates import ModalQuery, generate_slate, load_modal_queries

# The nine keys every candidate row must carry so the rows template renders.
_ROWS_KEYS = {
    "scene_id",
    "film_slug",
    "film_title",
    "year",
    "timecode",
    "description",
    "tags",
    "score",
    "keyframe_url",
}

# Path to a real keyframe so the image-validation branch has a file that
# actually exists on disk (validation requires ``image_path`` to exist).
_REAL_IMAGE = (
    "data/library/jeca_tatu/frames/scenes/keyframes_content/"
    "Mazzaropi-Jeca_Tatu_Paixão_Flix-Scene-012-02.jpg"
)


def _write_yaml(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def _three_type_yaml(image_path: str) -> str:
    """One representative entry per type, in the real m3_full shape."""
    return f"""\
dataset: m3_test
version: 1
queries:
  - id: text-01
    query_type: text
    text: "trabalhador rural arando o campo"
    lang: pt
    relevant_scene_ids: [34, 50, 110]
    relevance:
      "34": 2
      "50": 3
      "110": 2
    notes: "Rural labour composition."
  - id: image-01
    query_type: image
    text: "(image query) Jeca scene 12 — face in shadow"
    image_path: "{image_path}"
    lang: en
    notes: "Anchor frame for a face-with-textured-wall composition."
  - id: rhyme-01
    query_type: rhyme
    anchor: "jeca_tatu/12"
    lang: en
    notes: "Anchor 1 — early outdoor; rural setting."
"""


def test_load_modal_queries_validates_per_type(tmp_path: Path):
    """All real-shape types parse; malformed entries raise EvalError."""
    repo_root = Path(__file__).resolve().parents[1]
    real_image = repo_root / _REAL_IMAGE
    assert real_image.exists(), f"fixture image missing: {real_image}"

    good = _write_yaml(tmp_path / "good.yaml", _three_type_yaml(str(real_image)))
    queries = load_modal_queries(good)

    assert len(queries) == 3
    assert [q.query_type for q in queries] == ["text", "image", "rhyme"]
    by_id = {q.id: q for q in queries}
    assert by_id["text-01"].text == "trabalhador rural arando o campo"
    assert by_id["text-01"].relevant_scene_ids == (34, 50, 110)
    assert by_id["text-01"].relevance == {"34": 2.0, "50": 3.0, "110": 2.0}
    assert by_id["image-01"].image_path == real_image
    assert by_id["rhyme-01"].anchor == "jeca_tatu/12"
    assert by_id["rhyme-01"].text is None
    # Frozen dataclass — mutation must fail.
    with pytest.raises((AttributeError, Exception)):
        by_id["text-01"].text = "mutate"  # type: ignore[misc]

    # rhyme anchor with no slash -> EvalError
    bad_anchor = _write_yaml(
        tmp_path / "bad_anchor.yaml",
        """\
queries:
  - id: rhyme-bad
    query_type: rhyme
    anchor: "nofile"
    lang: en
""",
    )
    with pytest.raises(EvalError):
        load_modal_queries(bad_anchor)

    # image whose image_path does not exist -> EvalError
    bad_image = _write_yaml(
        tmp_path / "bad_image.yaml",
        """\
queries:
  - id: image-bad
    query_type: image
    image_path: "does/not/exist/frame.jpg"
    lang: en
""",
    )
    with pytest.raises(EvalError):
        load_modal_queries(bad_image)

    # unknown query_type -> EvalError
    bad_type = _write_yaml(
        tmp_path / "bad_type.yaml",
        """\
queries:
  - id: weird-01
    query_type: telepathy
    text: "x"
    lang: en
""",
    )
    with pytest.raises(EvalError):
        load_modal_queries(bad_type)


# ── generate_slate dispatch (hermetic) ──────────────────────────────────────


class _StubEmbedder:
    """Embedder stub: encode_text returns a fixed unit vector of dim ``d``."""

    def __init__(self, d: int = 4) -> None:
        v = np.zeros(d, dtype="float32")
        v[0] = 1.0
        self._v = v

    def encode_text(self, text: str) -> np.ndarray:  # noqa: D401 - stub
        return self._v


def _cfg() -> SimpleNamespace:
    """Minimal cfg; rhymes knobs read via getattr so this shape is enough."""
    return SimpleNamespace(
        retrieval=SimpleNamespace(rhymes=SimpleNamespace(diversity=0.5, k_candidates=30)),
    )


def test_generate_slate_dispatches_rhyme_to_find_rhymes(tmp_path: Path, monkeypatch):
    """Rhyme query → real find_rhymes (faked) → two 9-key rows."""
    import cinemateca.eval.slates as slates
    from cinemateca.rhymes.algorithm import Rhyme

    captured: dict[str, Any] = {}

    def _fake_find_rhymes(
        library_dir, anchor_slug, anchor_scene_id, top_n=8, cross_film_only=True, **kw
    ):
        captured["anchor_slug"] = anchor_slug
        captured["anchor_scene_id"] = anchor_scene_id
        captured["cross_film_only"] = cross_film_only
        captured["top_n"] = top_n
        return [
            Rhyme(
                film_slug="porter",
                scene_id=4,
                score=0.91,
            ),
            Rhyme(
                film_slug="porter",
                scene_id=2,
                score=0.80,
            ),
        ]

    monkeypatch.setattr(slates, "find_rhymes", _fake_find_rhymes)

    q = ModalQuery(
        id="rhyme-01",
        query_type="rhyme",
        text=None,
        image_path=None,
        anchor="jeca_tatu/12",
        w=None,
        lang="en",
        relevant_scene_ids=(),
        relevance={},
        notes=None,
    )
    rows = generate_slate(query=q, cfg=_cfg(), library_dir=tmp_path, k=2)

    assert len(rows) == 2
    for r in rows:
        assert set(r.keys()) == _ROWS_KEYS
    assert rows[0]["scene_id"] == 4
    assert rows[0]["film_slug"] == "porter"
    # Anchor parsed correctly and cross_film_only honoured.
    assert captured["anchor_slug"] == "jeca_tatu"
    assert captured["anchor_scene_id"] == 12
    assert captured["cross_film_only"] is True


def test_candidate_row_resolves_real_keyframe_url():
    """Generated slate rows must use the scene's real stored keyframe path,
    not a hardcoded /frames/scene_NNNN.jpg (review #4)."""
    from cinemateca.eval.slates import _candidate_row, _FilmMeta

    data_dir = Path("/srv/data").resolve()
    fp = (
        data_dir
        / "library"
        / "jeca"
        / "frames"
        / "scenes"
        / "keyframes_content"
        / "x-Scene-007-01.jpg"
    )
    meta = _FilmMeta(
        title="Jeca",
        year=1959,
        fps=24.0,
        kf_by_scene={7: {"scene_id": 7, "filepath": str(fp), "start_time_s": 12.0}},
        desc_by_scene={},
        tags_by_scene={},
        data_dir=data_dir,
    )
    row = _candidate_row(scene_id=7, film_slug="jeca", score=0.9, meta=meta)
    assert (
        row["keyframe_url"]
        == "/media/library/jeca/frames/scenes/keyframes_content/x-Scene-007-01.jpg"
    )


def test_candidate_row_missing_keyframe_falls_back_to_empty():
    """A scene with no on-disk keyframe yields keyframe_url='' (contract holds)."""
    from cinemateca.eval.slates import _candidate_row, _empty_meta

    meta = _empty_meta("jeca", Path("/srv/data").resolve())
    row = _candidate_row(scene_id=99, film_slug="jeca", score=0.5, meta=meta)
    assert row["keyframe_url"] == ""


def test_generate_slate_text_scopes_search_to_film_slug(tmp_path, monkeypatch):
    """#3: film_slug scopes the search to that film BEFORE top-k truncation, so a
    film that another film would crowd out of the global head still returns rows."""
    import cinemateca.eval.slates as slates
    from cinemateca.search.types import Hit, Query, SearchResult

    called: list[str] = []

    def fake_find(q: Query, *, film, mode, top_k, cfg):
        called.append(film.slug)
        return SearchResult(
            hits=[Hit(scene_id=1, score=0.9, keyframe_path="")],
            mode="clip",
            weights=None,
            query=q,
        )

    monkeypatch.setattr(slates, "find", fake_find)
    monkeypatch.setattr(slates, "_iter_films", lambda lib: ["alpha", "beta"])
    monkeypatch.setattr(slates, "_ctx_for", lambda lib, slug: SimpleNamespace(slug=slug))

    q = ModalQuery(
        id="t-01",
        query_type="text",
        text="rural",
        image_path=None,
        anchor=None,
        w=None,
        lang="en",
        relevant_scene_ids=(),
        relevance={},
        notes=None,
    )

    # Scoped: only "beta" is searched, so it can't be crowded out by "alpha".
    called.clear()
    rows = generate_slate(query=q, cfg=_cfg(), library_dir=tmp_path, k=5, film_slug="beta")
    assert called == ["beta"]
    assert rows and all(r["film_slug"] == "beta" for r in rows)

    # Unscoped (default): both films searched (the pre-fix global behaviour).
    called.clear()
    generate_slate(query=q, cfg=_cfg(), library_dir=tmp_path, k=5)
    assert called == ["alpha", "beta"]


# ── text / image find path (cross-film, registry-free) ──────────────────────


def _search_result(hits: list[Any]) -> Any:
    """Build a minimal CLIP-mode :class:`SearchResult` from a list of Hits."""
    from cinemateca.search.types import Query, SearchResult

    return SearchResult(hits=hits, mode="clip", weights=None, query=Query.of_text("q"))


def _hit(scene_id: int, score: float, film_slug: str) -> Any:
    from cinemateca.search.types import Hit

    return Hit(
        scene_id=scene_id,
        score=score,
        keyframe_path=f"/x/{film_slug}/frames/scene_{scene_id:04d}.jpg",
        film_slug=film_slug,
        description=f"desc for {film_slug}/{scene_id}",
    )


def test_generate_slate_dispatches_text_to_find(tmp_path: Path, monkeypatch):
    """Text query → CLIP ``find`` per film, merged by descending score.

    Exercises Fix #1: TWO on-disk film subdirs that are NOT registered in a
    ``films.json`` still yield rows (the disk-scan fallback in ``_iter_films``
    plus the path-derived ``_ctx_for`` must produce a real ``find`` context).
    """
    import cinemateca.eval.slates as slates

    (tmp_path / "film_a" / "embeddings").mkdir(parents=True)
    (tmp_path / "film_b" / "embeddings").mkdir(parents=True)

    captured: list[dict[str, Any]] = []

    def _fake_find(query, *, film, mode, top_k, cfg, **kw):
        # Capture per-call so we can assert the path-derived ctx + cross-film walk.
        captured.append({"slug": film.slug, "is_text": query.text is not None, "mode": mode})
        # Interleave scores across the two films so the global sort is observable.
        per_film = {
            "film_a": [_hit(10, 0.9, "film_a"), _hit(11, 0.5, "film_a")],
            "film_b": [_hit(20, 0.8, "film_b"), _hit(21, 0.4, "film_b")],
        }
        return _search_result(per_film[film.slug])

    monkeypatch.setattr(slates, "find", _fake_find)

    q = ModalQuery(
        id="text-01",
        query_type="text",
        text="trabalhador rural arando o campo",
        image_path=None,
        anchor=None,
        w=None,
        lang="pt",
        relevant_scene_ids=(),
        relevance={},
        notes=None,
    )
    rows = generate_slate(query=q, cfg=_cfg(), library_dir=tmp_path, k=9)

    # find called once per (unregistered) film, always in clip mode with text.
    assert {c["slug"] for c in captured} == {"film_a", "film_b"}
    assert all(c["mode"] == "clip" and c["is_text"] for c in captured)
    # Non-empty rows, all 9 keys, descending score, BOTH films present (merge).
    assert len(rows) == 4
    for r in rows:
        assert set(r.keys()) == _ROWS_KEYS
    scores = [r["score"] for r in rows]
    assert scores == sorted(scores, reverse=True)
    assert {r["film_slug"] for r in rows} == {"film_a", "film_b"}
    # Top hit is film_a/10 (0.9); the films interleave below it.
    assert (rows[0]["film_slug"], rows[0]["scene_id"]) == ("film_a", 10)
    assert (rows[1]["film_slug"], rows[1]["scene_id"]) == ("film_b", 20)


def test_generate_slate_dispatches_image_to_find(tmp_path: Path, monkeypatch):
    """Image query → CLIP ``find(Query.image(...))`` per film, registry-free."""
    import cinemateca.eval.slates as slates

    (tmp_path / "film_a" / "embeddings").mkdir(parents=True)
    img = tmp_path / "anchor.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")  # file must exist for validation

    captured: dict[str, Any] = {}

    def _fake_find(query, *, film, mode, top_k, cfg, **kw):
        captured["image_path"] = query.image_path
        captured["text"] = query.text
        captured["mode"] = mode
        return _search_result([_hit(3, 0.7, film.slug), _hit(8, 0.2, film.slug)])

    monkeypatch.setattr(slates, "find", _fake_find)

    q = ModalQuery(
        id="image-01",
        query_type="image",
        text=None,
        image_path=img,
        anchor=None,
        w=None,
        lang="en",
        relevant_scene_ids=(),
        relevance={},
        notes=None,
    )
    rows = generate_slate(query=q, cfg=_cfg(), library_dir=tmp_path, k=9)

    # find was called with an image Query (image_path set, text None) in clip mode.
    assert captured["image_path"] == img.resolve()
    assert captured["text"] is None
    assert captured["mode"] == "clip"
    assert len(rows) == 2
    for r in rows:
        assert set(r.keys()) == _ROWS_KEYS
    assert rows[0]["score"] >= rows[1]["score"]
