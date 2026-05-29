"""Tests for the non-destructive AI-tag correction (override) layer.

Covers: merge_tag_index suppression (normalisation-aware), the
tag_overrides.json load/save/set helpers, load_tag_index integration, and
BM25 cache invalidation on an override write.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from cinemateca.annotations import merge_tag_index
from cinemateca.annotations.overrides import (
    load as load_overrides,
)
from cinemateca.annotations.overrides import (
    save as save_overrides,
)
from cinemateca.annotations.overrides import (
    set_suppressed,
    suppressed_for_scene,
)
from cinemateca.library import FilmContext
from cinemateca.library.metadata import load_tag_index


def _make_ctx(tmp_path: Path) -> FilmContext:
    """Minimal per-film FilmContext rooted at a fresh metadata dir."""
    film_dir = tmp_path / "library" / "demo"
    metadata_dir = film_dir / "metadata"
    metadata_dir.mkdir(parents=True)
    return FilmContext(
        slug="demo",
        raw_path=film_dir / "raw",
        data_dir=tmp_path.resolve(),
        metadata_dir=metadata_dir,
        frames_dir=film_dir / "frames",
        embeddings_dir=film_dir / "embeddings",
    )


def test_merge_without_overrides_is_unchanged() -> None:
    """The default path (overrides=None) must be byte-identical to before."""
    llm = {"boat": [1, 2], "river": [2]}
    manual = {"3": ["Open Field"]}
    baseline = merge_tag_index(dict(llm), dict(manual))
    with_none = merge_tag_index(dict(llm), dict(manual), None)
    with_empty = merge_tag_index(dict(llm), dict(manual), {})
    assert baseline == {"boat": [1, 2], "river": [2], "open-field": ["3"]}
    assert with_none == baseline
    assert with_empty == baseline


def test_merge_suppresses_pair_and_drops_emptied_tag() -> None:
    llm = {"boat": [1, 2], "river": [2]}
    overrides = {"2": {"suppressed": ["boat", "river"]}}
    merged = merge_tag_index(llm, {}, overrides)
    # boat keeps scene 1, river had only scene 2 -> dropped entirely.
    assert merged == {"boat": [1]}


def test_suppression_matches_across_case_space_and_id_type() -> None:
    """Normalisation-aware on both axes: tag form and int/str scene id."""
    llm = {"night-time": [5]}  # merged key already canonical
    overrides = {"5": {"suppressed": ["Night Time"]}}  # raw curator input
    assert merge_tag_index(llm, {}, overrides) == {}

    # int scene id in the LLM list, str key in the overrides dict.
    llm2 = {"boat": [7]}
    assert merge_tag_index(llm2, {}, {"7": {"suppressed": ["boat"]}}) == {}


def test_overrides_roundtrip_and_set_helpers(tmp_path: Path) -> None:
    assert load_overrides(tmp_path) == {}

    ov: dict = {}
    set_suppressed(ov, 5, "Night Time", suppressed=True)
    set_suppressed(ov, 5, "night time", suppressed=True)  # dedupe via norm
    assert ov == {"5": {"suppressed": ["night-time"]}}
    assert suppressed_for_scene(ov, 5) == ["night-time"]
    assert suppressed_for_scene(ov, "5") == ["night-time"]

    save_overrides(tmp_path, ov)
    assert load_overrides(tmp_path) == ov

    # Removing the last suppressed tag prunes the scene entry entirely.
    set_suppressed(ov, 5, "night-time", suppressed=False)
    assert ov == {}


def test_load_tag_index_honors_overrides(tmp_path: Path) -> None:
    (tmp_path / "scene_tags.json").write_text(json.dumps({"boat": [1, 2], "river": [2]}))
    (tmp_path / "manual_annotations.json").write_text(json.dumps({}))
    assert load_tag_index(tmp_path) == {"boat": [1, 2], "river": [2]}

    (tmp_path / "tag_overrides.json").write_text(
        json.dumps({"2": {"suppressed": ["boat", "river"]}})
    )
    assert load_tag_index(tmp_path) == {"boat": [1]}


def test_bm25_cache_invalidates_on_override_write(tmp_path: Path) -> None:
    from cinemateca.search.bm25 import bm25_index_for_dir

    md = tmp_path / "metadata"
    md.mkdir()
    (md / "scene_descriptions.json").write_text(
        json.dumps(
            [
                {"scene_id": 0, "description": "barco no rio"},
                {"scene_id": 1, "description": "homem na rua"},
                {"scene_id": 2, "description": "carro na estrada"},
            ]
        )
    )
    (md / "scene_tags.json").write_text(json.dumps({"barco": [0]}))
    (md / "manual_annotations.json").write_text(json.dumps({}))

    a = bm25_index_for_dir(metadata_dir=md, stopwords_lang=None, k1=1.5, b=0.75)
    b = bm25_index_for_dir(metadata_dir=md, stopwords_lang=None, k1=1.5, b=0.75)
    assert a is b, "cache hit must return same object"

    time.sleep(0.01)
    (md / "tag_overrides.json").write_text(json.dumps({"0": {"suppressed": ["barco"]}}))
    c = bm25_index_for_dir(metadata_dir=md, stopwords_lang=None, k1=1.5, b=0.75)
    assert c is not a, "tag_overrides.json write must invalidate the BM25 cache"


# ── Service-layer curation helpers (delete / rename / suppress) ──────────────


def test_delete_manual_tag_removes_one_and_prunes(tmp_path: Path) -> None:
    from api.services.annotations import delete_manual_tag, load_annotations, save_annotations

    ctx = _make_ctx(tmp_path)
    save_annotations(ctx, {"5": ["rural", "exterior"], "6": ["solo"]})

    delete_manual_tag(ctx, 5, "Rural")  # normalisation-aware match
    assert load_annotations(ctx) == {"5": ["exterior"], "6": ["solo"]}

    delete_manual_tag(ctx, 6, "solo")  # last tag -> scene entry pruned
    assert load_annotations(ctx) == {"5": ["exterior"]}


def test_rename_manual_tag_normalises_and_dedupes(tmp_path: Path) -> None:
    from api.services.annotations import load_annotations, rename_manual_tag, save_annotations

    ctx = _make_ctx(tmp_path)
    save_annotations(ctx, {"5": ["rural", "exterior"]})
    rename_manual_tag(ctx, 5, "rural", "Open Field")
    assert load_annotations(ctx) == {"5": ["open-field", "exterior"]}

    # Renaming onto an existing tag collapses the duplicate.
    rename_manual_tag(ctx, 5, "open-field", "exterior")
    assert load_annotations(ctx) == {"5": ["exterior"]}


def test_rename_to_empty_falls_back_to_delete(tmp_path: Path) -> None:
    from api.services.annotations import load_annotations, rename_manual_tag, save_annotations

    ctx = _make_ctx(tmp_path)
    save_annotations(ctx, {"5": ["rural", "exterior"]})
    rename_manual_tag(ctx, 5, "rural", "   ")
    assert load_annotations(ctx) == {"5": ["exterior"]}


def test_toggle_ai_tag_writes_override_not_scene_tags(tmp_path: Path) -> None:
    import json

    from api.services.annotations import toggle_ai_tag

    ctx = _make_ctx(tmp_path)
    scene_tags = {"boat": [5], "river": [5]}
    (ctx.metadata_dir / "scene_tags.json").write_text(json.dumps(scene_tags))

    toggle_ai_tag(ctx, 5, "boat", suppressed=True)
    assert load_overrides(ctx.metadata_dir) == {"5": {"suppressed": ["boat"]}}
    # scene_tags.json (model output) is never mutated.
    assert json.loads((ctx.metadata_dir / "scene_tags.json").read_text()) == scene_tags
    # ... and the merged index drops the suppressed pair.
    assert load_tag_index(ctx.metadata_dir) == {"river": [5]}

    toggle_ai_tag(ctx, 5, "boat", suppressed=False)
    assert load_overrides(ctx.metadata_dir) == {}
    assert load_tag_index(ctx.metadata_dir) == {"boat": [5], "river": [5]}


def test_ai_tags_for_scene_reports_suppressed_state(tmp_path: Path) -> None:
    import json

    from cinemateca.annotations.scenes import ai_tags_for_scene

    ctx = _make_ctx(tmp_path)
    (ctx.metadata_dir / "scene_tags.json").write_text(
        json.dumps({"boat": [5, 6], "river": [5], "field": [6]})
    )
    (ctx.metadata_dir / "tag_overrides.json").write_text(
        json.dumps({"5": {"suppressed": ["boat"]}})
    )

    rows = ai_tags_for_scene(ctx, 5)
    assert rows == [
        {"tag": "boat", "suppressed": True},
        {"tag": "river", "suppressed": False},
    ]
    # scene 6 has no override -> nothing suppressed
    assert ai_tags_for_scene(ctx, 6) == [
        {"tag": "boat", "suppressed": False},
        {"tag": "field", "suppressed": False},
    ]
