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
