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
   hermetic: the backends (``load_audio_index`` / ``search_audio`` /
   ``find_rhymes``) are monkeypatched with fakes, and the cfg is a
   ``SimpleNamespace``. No real models or indexes are loaded.
"""

from __future__ import annotations

import json
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


def _five_type_yaml(image_path: str) -> str:
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
  - id: audio-01
    query_type: audio
    text: "música orquestral com cordas"
    lang: pt
    notes: "Orchestral string-section probe."
  - id: fusion-01
    query_type: fusion
    text: "silent landscape with melancholic music"
    w: 0.5
    lang: en
    notes: "Spec's canonical fusion example."
  - id: rhyme-01
    query_type: rhyme
    anchor: "jeca_tatu/12"
    lang: en
    notes: "Anchor 1 — early outdoor; rural setting."
"""


def test_load_modal_queries_validates_per_type(tmp_path: Path):
    """All five real-shape types parse; malformed entries raise EvalError."""
    repo_root = Path(__file__).resolve().parents[1]
    real_image = repo_root / _REAL_IMAGE
    assert real_image.exists(), f"fixture image missing: {real_image}"

    good = _write_yaml(tmp_path / "good.yaml", _five_type_yaml(str(real_image)))
    queries = load_modal_queries(good)

    assert len(queries) == 5
    assert [q.query_type for q in queries] == ["text", "image", "audio", "fusion", "rhyme"]
    by_id = {q.id: q for q in queries}
    assert by_id["text-01"].text == "trabalhador rural arando o campo"
    assert by_id["text-01"].relevant_scene_ids == (34, 50, 110)
    assert by_id["text-01"].relevance == {"34": 2.0, "50": 3.0, "110": 2.0}
    assert by_id["image-01"].image_path == real_image
    assert by_id["fusion-01"].w == 0.5
    assert by_id["rhyme-01"].anchor == "jeca_tatu/12"
    assert by_id["rhyme-01"].text is None
    # Frozen dataclass — mutation must fail.
    with pytest.raises((AttributeError, Exception)):
        by_id["text-01"].text = "mutate"  # type: ignore[misc]

    # fusion w out of range -> EvalError
    bad_w = _write_yaml(
        tmp_path / "bad_w.yaml",
        """\
queries:
  - id: fusion-bad
    query_type: fusion
    text: "x"
    w: 1.5
    lang: en
""",
    )
    with pytest.raises(EvalError):
        load_modal_queries(bad_w)

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


def _fake_audio_index() -> Any:
    """Two-row fake AudioIndex: row 0 aligns with the stub query vector."""
    from cinemateca.search.audio import AudioIndex

    emb = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype="float32")
    mapping = [
        {"scene_id": 7, "wav_path": "", "start_time_s": None, "end_time_s": None},
        {"scene_id": 3, "wav_path": "", "start_time_s": None, "end_time_s": None},
    ]
    return AudioIndex(embeddings=emb, mapping=mapping)


def _cfg() -> SimpleNamespace:
    """Minimal cfg; rhymes knobs read via getattr so this shape is enough."""
    return SimpleNamespace(
        retrieval=SimpleNamespace(rhymes=SimpleNamespace(diversity=0.5, k_candidates=30)),
    )


def test_generate_slate_dispatches_audio_to_search_audio(tmp_path: Path, monkeypatch):
    """Audio query → real search_audio over a fake CLAP index → 9-key rows."""
    import cinemateca.eval.slates as slates

    # One physically-present film so the film-walk has a slug to iterate;
    # the patched load_audio_index ignores the path and returns the fake.
    (tmp_path / "jeca").mkdir()

    monkeypatch.setattr(slates, "load_audio_index", lambda audio_dir: _fake_audio_index())
    monkeypatch.setattr(slates, "get_audio_embedder", lambda cfg, device=None: _StubEmbedder(4))

    q = ModalQuery(
        id="audio-01",
        query_type="audio",
        text="música orquestral com cordas",
        image_path=None,
        anchor=None,
        w=None,
        lang="pt",
        relevant_scene_ids=(),
        relevance={},
        notes=None,
    )
    rows = generate_slate(query=q, cfg=_cfg(), library_dir=tmp_path, k=2)

    assert isinstance(rows, list)
    assert len(rows) == 2
    for r in rows:
        assert isinstance(r, dict)
        assert set(r.keys()) == _ROWS_KEYS
    # Ordered by descending score (row 0 of the fake index scores 1.0).
    assert rows[0]["score"] >= rows[1]["score"]
    assert rows[0]["scene_id"] == 7


def test_generate_slate_audio_empty_when_no_index(tmp_path: Path, monkeypatch):
    """A film with no CLAP index (load_audio_index -> None) yields an empty slate."""
    import cinemateca.eval.slates as slates

    (tmp_path / "jeca").mkdir()
    monkeypatch.setattr(slates, "load_audio_index", lambda audio_dir: None)
    monkeypatch.setattr(slates, "get_audio_embedder", lambda cfg, device=None: _StubEmbedder(4))

    q = ModalQuery(
        id="audio-01",
        query_type="audio",
        text="x",
        image_path=None,
        anchor=None,
        w=None,
        lang="pt",
        relevant_scene_ids=(),
        relevance={},
        notes=None,
    )
    assert generate_slate(query=q, cfg=_cfg(), library_dir=tmp_path, k=2) == []


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
                keyframe_path=Path("/x/porter/frames/scene_0004.jpg"),
            ),
            Rhyme(
                film_slug="porter",
                scene_id=2,
                score=0.80,
                keyframe_path=Path("/x/porter/frames/scene_0002.jpg"),
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


# ── fusion index-loading + _normalise_clip_mapping ──────────────────────────


def test_slate_fusion_loads_indexes_and_calls_search_fusion(tmp_path: Path, monkeypatch):
    """Fusion query → on-disk CLIP index loaded + ``search_fusion`` called → 9-key rows.

    The CLIP index files are written to disk so the real ``np.load`` +
    ``_normalise_clip_mapping`` path runs; CLAP is absent (``load_audio_index``
    → None), so the CLIP-only fusion branch is exercised. ``search_fusion`` is
    faked to return the canonical 4-key hit dicts.
    """
    import cinemateca.eval.slates as slates

    emb_dir = tmp_path / "jeca" / "embeddings"
    emb_dir.mkdir(parents=True)
    np.save(emb_dir / "keyframe_embeddings.npy", np.eye(3, dtype="float32"))
    # Parallel-array mapping shape (the SigLIP2 writer) — must be normalised.
    (emb_dir / "index_mapping.json").write_text(
        json.dumps({"scene_ids": [5, 6, 7], "total_vectors": 3}), encoding="utf-8"
    )

    seen: dict[str, Any] = {}

    def _fake_search_fusion(*, clip_emb, clap_emb, clip_mapping, clap_mapping, **kw):
        # Assert the on-disk CLIP index reached the searcher in normalised form.
        seen["clip_emb_shape"] = clip_emb.shape
        seen["clip_mapping"] = clip_mapping
        return [
            {"scene_id": 5, "score": 0.88, "clip_score": 0.88, "clap_score": 0.0},
            {"scene_id": 7, "score": 0.42, "clip_score": 0.42, "clap_score": 0.0},
        ]

    monkeypatch.setattr(slates, "load_audio_index", lambda audio_dir: None)
    monkeypatch.setattr(slates, "get_image_embedder", lambda cfg, device=None: _StubEmbedder(4))
    monkeypatch.setattr(slates, "get_audio_embedder", lambda cfg, device=None: _StubEmbedder(4))
    monkeypatch.setattr(slates, "search_fusion", _fake_search_fusion)

    q = ModalQuery(
        id="fusion-01",
        query_type="fusion",
        text="silent landscape with melancholic music",
        image_path=None,
        anchor=None,
        w=0.5,
        lang="en",
        relevant_scene_ids=(),
        relevance={},
        notes=None,
    )
    rows = generate_slate(query=q, cfg=_cfg(), library_dir=tmp_path, k=9)

    assert len(rows) == 2
    for r in rows:
        assert set(r.keys()) == _ROWS_KEYS
    assert rows[0]["score"] >= rows[1]["score"]
    assert rows[0]["scene_id"] == 5
    # The real np.load + _normalise_clip_mapping fed search_fusion correctly.
    assert seen["clip_emb_shape"] == (3, 3)
    assert seen["clip_mapping"] == [{"scene_id": 5}, {"scene_id": 6}, {"scene_id": 7}]


def test_normalise_clip_mapping_handles_both_shapes():
    """``_normalise_clip_mapping`` accepts BOTH the parallel-array and list shapes."""
    from cinemateca.eval.slates import _normalise_clip_mapping

    # Parallel-array shape (SigLIP2 writer): {"scene_ids": [...]}.
    parallel = _normalise_clip_mapping({"scene_ids": [5, 6, 7], "total_vectors": 3})
    assert parallel == [{"scene_id": 5}, {"scene_id": 6}, {"scene_id": 7}]

    # List-of-dicts shape: [{"scene_id": ...}, ...] (extra keys ignored).
    listish = _normalise_clip_mapping(
        [{"scene_id": 9, "filepath": "a.jpg"}, {"scene_id": 4, "filepath": "b.jpg"}]
    )
    assert listish == [{"scene_id": 9}, {"scene_id": 4}]

    # An unrecognised shape raises EvalError (defensive, E3b-critical).
    with pytest.raises(EvalError):
        _normalise_clip_mapping(42)
