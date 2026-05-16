"""
tests/test_scene_id_filtering.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 1c of the FastAPI regression-recovery effort.

Bug: tag filtering silently returns empty / wrong results when scene IDs
have mixed int/str types.

  * ``LLMDescriber.build_tag_index`` stores scene_id values as **ints**
    (records use ``int(row.get("scene_id", -1))``). Round-tripped through
    JSON they stay Python ints (list VALUES, not object keys).
  * Manual annotations (``annotator.load``) are a JSON object, so keys are
    **strings**. ``merge_tag_index`` produces ONE hybrid index whose value
    lists mix ints (LLM) and strs (manual).
  * ``scenes._build_cards`` does ``set(tag_index.get(tag, []))`` then
    ``str(s["scene_id"]) in valid_ids`` — ``"351" in {351}`` is False, so
    LLM-tag filters return nothing.
  * ``SemanticSearch.combined`` does ``df["scene_id"].isin(valid_ids)`` —
    the df column is int; str manual ids never match.

These tests pin the canonical-key helpers and both filter sites. They are
hermetic: no GPU / CLIP / network / real video. The Search seam is unit
tested directly against ``SemanticSearch.combined`` with a hand-built
keyframes_df + a stub embedder (the tag-filter masking happens BEFORE any
CLIP call, and ``encode_text`` is the only model touch — stubbed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Unit tests: the canonical-key helpers ─────────────────────────────────────

class TestSceneIdKey:
    def test_int(self):
        from cinemateca.scene_ids import scene_id_key

        assert scene_id_key(351) == "351"

    def test_str(self):
        from cinemateca.scene_ids import scene_id_key

        assert scene_id_key("351") == "351"

    def test_numpy_int(self):
        from cinemateca.scene_ids import scene_id_key

        assert scene_id_key(np.int64(351)) == "351"

    def test_integral_float_strips_trailing_zero(self):
        from cinemateca.scene_ids import scene_id_key

        # pandas/JSON can yield 351.0 for an integer column with NaNs.
        assert scene_id_key(351.0) == "351"
        assert scene_id_key(np.float64(351.0)) == "351"

    def test_whitespace_stripped(self):
        from cinemateca.scene_ids import scene_id_key

        assert scene_id_key("  351 ") == "351"

    def test_normalize_tag_index_mixed(self):
        from cinemateca.scene_ids import normalize_tag_index

        # Mixed int (LLM) + str (manual) values, as merge_tag_index yields.
        hybrid = {"exterior": [351, 352, "353"], "dia": ["351"]}
        norm = normalize_tag_index(hybrid)
        assert norm == {"exterior": {"351", "352", "353"}, "dia": {"351"}}
        for v in norm.values():
            assert all(isinstance(x, str) for x in v)

    def test_normalize_scene_record(self):
        from cinemateca.scene_ids import normalize_scene_record

        rec = {"scene_id": 351, "tags": ["a"]}
        out = normalize_scene_record(rec)
        assert out["scene_id"] == "351"
        assert out["tags"] == ["a"]
        # Original not mutated.
        assert rec["scene_id"] == 351


# ── Shared fixture: isolated client with seeded metadata ──────────────────────

@pytest.fixture()
def seeded_client(tmp_path, monkeypatch):
    """TestClient with temp config and metadata files seeded for tag tests."""
    from cinemateca.config import load_config

    cfg = load_config(project_root=tmp_path)
    for name in (
        "data_dir", "raw_dir", "frames_dir", "metadata_dir",
        "embeddings_dir", "models_dir", "outputs_dir", "logs_dir",
    ):
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        setattr(cfg.paths, name, d)

    meta_dir = Path(cfg.paths.metadata_dir)

    # Three keyframes / scenes. No timecode_start so the grid template
    # falls back to rendering "scene <id>" — a precise per-scene marker
    # (the stored filepath is never emitted; only the /media img_url is,
    # and these paths resolve outside data_dir so img_url is None).
    kf_meta = [
        {"scene_id": 351, "filepath": "frames/s351.jpg"},
        {"scene_id": 352, "filepath": "frames/s352.jpg"},
        {"scene_id": 353, "filepath": "frames/s353.jpg"},
    ]
    (meta_dir / "keyframes_metadata.json").write_text(json.dumps(kf_meta))

    # LLM scene_tags.json — built by LLMDescriber.build_tag_index, so values
    # are INTS round-tripped through JSON.
    llm_tags = {"exterior": [351, 352], "dia": [351]}
    (meta_dir / "scene_tags.json").write_text(json.dumps(llm_tags))

    # Manual annotations — JSON object => STRING keys.
    manual = {"353": ["manual-only"], "351": ["exterior"]}
    (meta_dir / "manual_annotations.json").write_text(json.dumps(manual))

    descriptions = [{"scene_id": s, "description": f"scene {s}"} for s in (351, 352, 353)]
    (meta_dir / "scene_descriptions.json").write_text(json.dumps(descriptions))

    import api.deps as deps

    deps.get_config.cache_clear()
    monkeypatch.setattr(deps, "get_config", lambda: cfg)

    import api.server as server
    from api.routes import annotate, library, processing, scenes, search

    for mod in (server, scenes, search, annotate, processing, library):
        if hasattr(mod, "get_config"):
            monkeypatch.setattr(mod, "get_config", lambda: cfg)

    import api.jobs as jobs

    monkeypatch.setattr(jobs, "_jobs", {})

    from api.server import app

    with TestClient(app) as c:
        c.cookies.set("locale", "en")
        yield c


# ── Scenes tab: tag-filter correctness ────────────────────────────────────────

class TestScenesTagFilter:
    @staticmethod
    def _scene_ids(html: str) -> set[str]:
        """Scene ids rendered as 'scene <id>' in the timecode fallback."""
        import re

        return set(re.findall(r"scene (\d+)", html))

    def test_llm_only_int_tag(self, seeded_client):
        """`exterior` LLM tag has int ids 351,352 (plus manual str 351).
        Must return scenes 351 and 352. Pre-fix: int ids never matched
        ``str(scene_id) in valid_ids`` so this returned the wrong set."""
        r = seeded_client.get("/api/scenes", params={"tags": ["exterior"]})
        assert r.status_code == 200, r.text[:500]
        assert self._scene_ids(r.text) == {"351", "352"}

    def test_manual_only_str_tag(self, seeded_client):
        """`manual-only` is a manual annotation (str id '353'). Pre-fix the
        kf_meta int scene_id 353 never matched str '353' in valid_ids."""
        r = seeded_client.get("/api/scenes", params={"tags": ["manual-only"]})
        assert r.status_code == 200, r.text[:500]
        assert self._scene_ids(r.text) == {"353"}

    def test_mixed_merged_tag(self, seeded_client):
        """`exterior` exists in BOTH LLM (ints 351,352) and manual
        (str '351'). The merged value list is [351,352,'351'] — mixed.
        Result must be scenes 351 and 352 (deduped by canonical key)."""
        r = seeded_client.get("/api/scenes", params={"tags": ["exterior"]})
        assert r.status_code == 200, r.text[:500]
        assert self._scene_ids(r.text) == {"351", "352"}


# ── Search: SemanticSearch.combined tag-filter masking ────────────────────────

class _StubEmbedder:
    """Replaces CLIPEmbedder for the combined() tag-filter path.

    ``combined()`` calls ``encode_text`` ONCE, only after masking. The
    mask (the bug site) is computed purely from the df + tag_index, so a
    deterministic fake text embedding is sufficient and CLIP-free.
    """

    def encode_text(self, query: str) -> np.ndarray:
        return np.ones(4, dtype="float32")


def _make_searcher(scene_ids):
    """Build a SemanticSearch over a tiny df with the given scene_ids."""
    from cinemateca.embeddings import SemanticSearch

    n = len(scene_ids)
    embeddings = np.eye(4, dtype="float32")[:n] if n <= 4 else np.ones((n, 4), "float32")
    kf_df = pd.DataFrame({
        "filepath": [f"frames/s{sid}.jpg" for sid in scene_ids],
        "scene_id": scene_ids,
    })
    return SemanticSearch(embeddings, kf_df, _StubEmbedder())


class TestCombinedTagFilter:
    def test_int_only_llm_tag(self):
        """df scene_id column is int; tag_index has int LLM ids."""
        searcher = _make_searcher([351, 352, 353])
        tag_index = {"exterior": [351, 352]}
        res = searcher.combined("x", ["exterior"], tag_index, top_k=8)
        assert set(res["scene_id"].map(str)) == {"351", "352"}

    def test_str_only_manual_tag(self):
        """Manual tag ids are strings; pre-fix ``df.isin({'353'})`` against
        an int column matched nothing."""
        searcher = _make_searcher([351, 352, 353])
        tag_index = {"manual-only": ["353"]}
        res = searcher.combined("x", ["manual-only"], tag_index, top_k=8)
        assert set(res["scene_id"].map(str)) == {"353"}

    def test_mixed_merged_tag(self):
        """Hybrid value list [351, 352, '351'] (LLM ints + manual str)."""
        searcher = _make_searcher([351, 352, 353])
        tag_index = {"exterior": [351, 352, "351"]}
        res = searcher.combined("x", ["exterior"], tag_index, top_k=8)
        assert set(res["scene_id"].map(str)) == {"351", "352"}
