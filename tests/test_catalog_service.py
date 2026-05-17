"""
tests/test_catalog_service.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 3a: direct unit tests for the extracted catalog service and
``FilmContext``.

These are NEW units (the service did not exist before Phase 3a), so
they ADD coverage on top of the Phase 0/1/2 route regression net —
they do not replace it. They pin the service's public surface
directly (no HTTP round-trip): ``FilmContext`` path resolution, the
shared ``load_json`` / ``keyframe_url`` primitives, metadata loading
on empty vs seeded dirs (incl. the int/str scene-id canonicalization
the catalog inherits from ``cinemateca.scene_ids``), card
construction + tag/keyword filtering, and the tab context builders.

Hermetic: built on the shared ``tmp_config`` / ``seed_metadata``
factory fixtures from conftest.py (no GPU, no CLIP, no real video,
no repo ``data/`` access — enforced by tmp_config's path guard).
"""

from __future__ import annotations

import json
from pathlib import Path

from api.services.catalog import (
    build_cards,
    build_scenes_context,
    build_scenes_grid,
    keyframe_url,
    load_json,
    load_metadata,
    load_tag_index,
)
from api.services.film_context import FilmContext

# ── FilmContext ───────────────────────────────────────────────────────────────

class TestFilmContext:
    def test_from_config_resolves_all_paths(self, tmp_config):
        ctx = FilmContext.from_config(tmp_config)
        assert ctx.slug is None  # Phase 3a is global/flat only
        assert ctx.raw_path == Path(tmp_config.paths.raw_dir)
        assert ctx.metadata_dir == Path(tmp_config.paths.metadata_dir)
        assert ctx.frames_dir == Path(tmp_config.paths.frames_dir)
        assert ctx.embeddings_dir == Path(tmp_config.paths.embeddings_dir)

    def test_data_dir_is_resolved(self, tmp_config):
        """data_dir must be ``.resolve()``-d — keyframe-URL math and the
        /media mount both compared resolved paths pre-extraction."""
        ctx = FilmContext.from_config(tmp_config)
        assert ctx.data_dir == Path(tmp_config.paths.data_dir).resolve()
        assert ctx.data_dir.is_absolute()

    def test_is_frozen(self, tmp_config):
        import dataclasses

        ctx = FilmContext.from_config(tmp_config)
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            ctx.slug = "x"  # type: ignore[misc]


# ── Shared primitives ─────────────────────────────────────────────────────────

class TestLoadJson:
    def test_missing_returns_none(self, tmp_path):
        assert load_json(tmp_path / "nope.json") is None

    def test_reads_list_and_dict(self, tmp_path):
        lp = tmp_path / "l.json"
        lp.write_text(json.dumps([1, 2]))
        dp = tmp_path / "d.json"
        dp.write_text(json.dumps({"a": 1}))
        assert load_json(lp) == [1, 2]
        assert load_json(dp) == {"a": 1}


class TestKeyframeUrl:
    def test_inside_data_dir_returns_media_url(self, tmp_path):
        data_dir = tmp_path / "data"
        (data_dir / "frames").mkdir(parents=True)
        kf = data_dir / "frames" / "s1.jpg"
        kf.touch()
        assert keyframe_url(kf, data_dir) == "/media/frames/s1.jpg"

    def test_outside_data_dir_returns_none(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        outside = tmp_path / "elsewhere" / "x.jpg"
        outside.parent.mkdir()
        outside.touch()
        assert keyframe_url(outside, data_dir) is None

    def test_accepts_str_and_path(self, tmp_path):
        """The unified primitive replaced two copies with str vs Path
        signatures — both inputs must behave identically."""
        data_dir = tmp_path / "data"
        (data_dir / "f").mkdir(parents=True)
        kf = data_dir / "f" / "k.jpg"
        kf.touch()
        assert keyframe_url(str(kf), data_dir) == keyframe_url(kf, data_dir)


# ── Metadata loading ──────────────────────────────────────────────────────────

class TestLoadMetadata:
    def test_empty_dir_yields_empty_structures(self, tmp_config):
        ctx = FilmContext.from_config(tmp_config)
        kf, desc, vis, tags = load_metadata(ctx.metadata_dir)
        assert kf == []
        assert desc == {}
        assert vis == {}
        assert tags == {}

    def test_seeded_default_dataset(self, tmp_config, seed_metadata):
        seed_metadata()
        ctx = FilmContext.from_config(tmp_config)
        kf, desc, vis, tags = load_metadata(ctx.metadata_dir)
        assert len(kf) == 2
        # desc / vis keyed by canonical STR scene id.
        assert set(desc) == {"351", "352"}
        assert set(vis) == {"351"}
        # tag_index normalized: STR ids, manual-only merged in.
        assert tags["exterior"] == {"351", "352"}
        assert tags["dia"] == {"351"}
        assert tags["manual-only"] == {"352"}
        assert all(
            isinstance(i, str) for ids in tags.values() for i in ids
        )

    def test_load_tag_index_raw_is_unnormalized(self, tmp_config, seed_metadata):
        """The search primitive returns the RAW merged index (mixed int
        LLM ids + str manual keys) — Phase 1c normalization happens in
        SemanticSearch.combined, not here. Only the keys feed
        available_tags."""
        seed_metadata()
        ctx = FilmContext.from_config(tmp_config)
        raw = load_tag_index(ctx.metadata_dir)
        # LLM values stay INT (un-normalized), proving it is not the
        # normalized index.
        assert 351 in raw["exterior"] and 352 in raw["exterior"]
        # Manual-only tag merged in (str key "352").
        assert "352" in raw["manual-only"]
        # Keys identical to the normalized index → available_tags same.
        norm_keys = sorted(load_metadata(ctx.metadata_dir)[3].keys())
        assert sorted(raw.keys()) == norm_keys


# ── Card construction ─────────────────────────────────────────────────────────

class TestBuildCards:
    def _load(self, tmp_config, seed_metadata):
        seed_metadata()
        ctx = FilmContext.from_config(tmp_config)
        return ctx, load_metadata(ctx.metadata_dir)

    def test_no_filter_returns_all(self, tmp_config, seed_metadata):
        ctx, (kf, desc, vis, tags) = self._load(tmp_config, seed_metadata)
        cards = build_cards(kf, desc, vis, tags, ctx.data_dir, [], "")
        assert {c["scene_id"] for c in cards} == {351, 352}

    def test_llm_int_tag_filter(self, tmp_config, seed_metadata):
        """`dia` is an LLM tag (int id 351). Canonical-key matching must
        still select scene 351 only."""
        ctx, (kf, desc, vis, tags) = self._load(tmp_config, seed_metadata)
        cards = build_cards(kf, desc, vis, tags, ctx.data_dir, ["dia"], "")
        assert [c["scene_id"] for c in cards] == [351]

    def test_manual_str_tag_filter(self, tmp_config, seed_metadata):
        ctx, (kf, desc, vis, tags) = self._load(tmp_config, seed_metadata)
        cards = build_cards(
            kf, desc, vis, tags, ctx.data_dir, ["manual-only"], ""
        )
        assert [c["scene_id"] for c in cards] == [352]

    def test_keyword_filter_on_description_blob(self, tmp_config, seed_metadata):
        ctx, (kf, desc, vis, tags) = self._load(tmp_config, seed_metadata)
        cards = build_cards(kf, desc, vis, tags, ctx.data_dir, [], "office")
        assert [c["scene_id"] for c in cards] == [352]

    def test_card_shape_and_truncation(self, tmp_config, seed_metadata):
        """Card keys/values match the template contract; description is
        capped at 120 chars and tags at 16 (frequency-sampled)."""
        seed_metadata(
            descriptions=[{"scene_id": 351, "description": "x" * 200}],
            llm_tags={f"t{i}": [351] for i in range(20)},
            manual=None,
            visual=[
                {
                    "scene_id": 351,
                    "environment": {
                        "location": "exterior",
                        "time_of_day": "dia",
                    },
                    "num_faces": 3,
                }
            ],
            scenes=[
                {
                    "scene_id": 351,
                    "filepath": "frames/s351.jpg",
                    "timecode_start": "00:01:23",
                }
            ],
        )
        ctx = FilmContext.from_config(tmp_config)
        kf, desc, vis, tags = load_metadata(ctx.metadata_dir)
        cards = build_cards(kf, desc, vis, tags, ctx.data_dir, [], "")
        c = cards[0]
        assert set(c) == {
            "scene_id",
            "img_url",
            "timecode",
            "tags",
            "environment",
            "num_people",
            "description",
        }
        assert len(c["description"]) == 120
        assert len(c["tags"]) == 16  # capped from 20 via frequency sampling
        assert c["environment"] == "exterior · dia"
        assert c["num_people"] == 3
        assert c["timecode"] == "00:01:23"


# ── Tab context builders ──────────────────────────────────────────────────────

class TestSceneContextBuilders:
    def test_scenes_context_empty(self, tmp_config):
        ctx = FilmContext.from_config(tmp_config)
        out = build_scenes_context(ctx)
        assert out == {"cards": [], "available_tags": [], "no_data": True}

    def test_scenes_context_seeded(self, tmp_config, seed_metadata):
        seed_metadata()
        ctx = FilmContext.from_config(tmp_config)
        out = build_scenes_context(ctx)
        assert out["no_data"] is False
        assert len(out["cards"]) == 2
        # available_tags = sorted union of LLM + manual tag keys.
        assert out["available_tags"] == sorted(
            ["exterior", "dia", "manual-only", "noite"]
        )

    def test_scenes_grid_filtered(self, tmp_config, seed_metadata):
        seed_metadata()
        ctx = FilmContext.from_config(tmp_config)
        out = build_scenes_grid(ctx, ["dia"], "")
        assert set(out) == {"cards"}
        assert [c["scene_id"] for c in out["cards"]] == [351]
