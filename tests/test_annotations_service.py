"""
tests/test_annotations_service.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 3b: direct unit tests for the extracted annotations service
(``api/services/annotations.py``) and the now-atomic
``cinemateca.annotator.save``.

These are NEW units (the service did not exist before Phase 3b), so
they ADD coverage on top of the Phase 0/1/2 route regression net and
the Phase-3a catalog units — they do not replace them. They pin the
service's public surface directly (no HTTP round-trip):

  * ``normalize_tags`` edge cases (case, internal-space→hyphen, empty
    fragments, whitespace, NO dedupe — byte-identical to the prior
    inline route list-comp the Phase-2 on-disk test pins);
  * ``save_annotations`` is atomic — content is byte-identical to the
    old plain rewrite AND no stray temp file is left in the metadata
    dir (the crash-safety contract: a same-dir temp + ``os.replace``);
  * ``load_annotations`` round-trip incl. missing-file → ``{}``;
  * ``build_scene_list`` ``no_llm`` vs ``all`` filtering (incl. the
    BROKEN-LLM placeholder rule);
  * ``scene_context`` / ``build_scene_panel`` default-to-first,
    jump-by-id, prev/next edges, ``annotated_count``;
  * ``build_annotate_context`` no_data / all_done / populated branches.

Hermetic: built on the shared ``tmp_config`` / ``seed_metadata``
factory fixtures from conftest.py (no GPU, no CLIP, no real video, no
repo ``data/`` access — enforced by tmp_config's path guard).
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from api.services.annotations import (
    _BROKEN_LLM,
    build_annotate_context,
    build_scene_list,
    build_scene_panel,
    load_annotations,
    normalize_tags,
    save_annotations,
    save_description,
    scene_context,
)
from api.services.film_context import FilmContext
from cinemateca.annotator import FILENAME

# ── normalize_tags ────────────────────────────────────────────────────────────

class TestNormalizeTags:
    def test_lowercase_and_space_to_hyphen(self):
        assert normalize_tags("Rural, Open Field") == ["rural", "open-field"]

    def test_empty_fragments_and_whitespace_dropped(self):
        # ",," and a trailing comma + a whitespace-only fragment must all
        # be dropped, exactly as the prior `if t.strip()` route list-comp.
        assert normalize_tags("Rural,,  , exterior,") == ["rural", "exterior"]

    def test_empty_string_yields_empty_list(self):
        assert normalize_tags("") == []
        assert normalize_tags("   ") == []
        assert normalize_tags(",,,") == []

    def test_order_preserved_and_no_dedupe(self):
        # The pre-extraction route did NOT dedupe; byte-identical
        # behaviour must be preserved (dedupe would be a behaviour
        # change, out of scope for a refactor phase).
        assert normalize_tags("b, a, b") == ["b", "a", "b"]

    def test_matches_legacy_inline_listcomp(self):
        """Property check vs the exact prior inline expression."""
        for raw in ["Rural,, Open Field , exterior,", "  A B  ,c", "", "x"]:
            legacy = [
                t.strip().lower().replace(" ", "-")
                for t in raw.split(",")
                if t.strip()
            ]
            assert normalize_tags(raw) == legacy


# ── load / save (atomic) ──────────────────────────────────────────────────────

class TestLoadSaveAnnotations:
    def test_load_missing_returns_empty_dict(self, tmp_config):
        ctx = FilmContext.from_config(tmp_config)
        assert load_annotations(ctx) == {}

    def test_save_then_load_roundtrip(self, tmp_config):
        ctx = FilmContext.from_config(tmp_config)
        data = {"351": ["rural", "exterior"], "352": ["noite"]}
        save_annotations(ctx, data)
        assert load_annotations(ctx) == data

    def test_save_writes_byte_identical_json(self, tmp_config):
        """Atomic save must produce the SAME bytes the old plain
        ``json.dump(..., indent=2, ensure_ascii=False)`` rewrite did —
        existing on-disk-JSON test assertions depend on this format."""
        ctx = FilmContext.from_config(tmp_config)
        data = {"351": ["açaí", "rural"]}  # non-ASCII pins ensure_ascii=False
        save_annotations(ctx, data)

        on_disk = (ctx.metadata_dir / FILENAME).read_text(encoding="utf-8")
        expected = json.dumps(data, indent=2, ensure_ascii=False)
        assert on_disk == expected
        # ensure_ascii=False: the literal multibyte char is present.
        assert "açaí" in on_disk

    def test_save_leaves_no_temp_file_behind(self, tmp_config):
        """Crash-safety contract part 1: the same-dir temp file used for
        the atomic ``os.replace`` must NOT survive a successful save —
        only ``manual_annotations.json`` should exist afterwards."""
        ctx = FilmContext.from_config(tmp_config)
        save_annotations(ctx, {"1": ["a"]})

        entries = sorted(p.name for p in Path(ctx.metadata_dir).iterdir())
        assert entries == [FILENAME]
        # Defensive: explicitly assert no leftover .tmp sibling.
        assert not list(Path(ctx.metadata_dir).glob(f".{FILENAME}.*.tmp"))

    def test_save_is_atomic_replace_not_truncate(self, tmp_config):
        """Crash-safety contract part 2: a serialization failure must
        leave the PREVIOUS complete file intact (os.replace happens only
        after the temp file is fully written) and leave no temp file."""
        ctx = FilmContext.from_config(tmp_config)
        good = {"351": ["kept"]}
        save_annotations(ctx, good)

        class Unserializable:
            pass

        with pytest.raises(TypeError):
            save_annotations(ctx, {"352": [Unserializable()]})  # type: ignore[list-item]

        # Old file untouched (not truncated/half-written) ...
        assert load_annotations(ctx) == good
        # ... and the failed write left no stray temp file.
        assert not list(Path(ctx.metadata_dir).glob(f".{FILENAME}.*.tmp"))

    def test_save_preserves_existing_file_mode(self, tmp_config):
        """Permission contract: an atomic save over an EXISTING file must
        keep that file's mode. mkstemp() makes the temp file 0600 and
        os.replace() moves the temp inode over the target, so without the
        chmod-match the first save would silently downgrade the file."""
        ctx = FilmContext.from_config(tmp_config)
        path = ctx.metadata_dir / FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        # Pre-create the target with a known NON-default mode.
        path.write_text(
            json.dumps({"351": ["old"]}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.chmod(path, 0o640)
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o640

        new_data = {"351": ["rural"], "352": ["noite"]}
        save_annotations(ctx, new_data)

        # Mode preserved ...
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o640
        # ... and the content was actually updated (not just mode kept).
        assert load_annotations(ctx) == new_data

    def test_save_new_file_uses_umask_default_mode(self, tmp_config):
        """Permission contract: a brand-new file must get the umask
        default (0o666 & ~umask) a plain open()-rewrite would produce,
        NOT mkstemp's hardcoded 0600."""
        ctx = FilmContext.from_config(tmp_config)
        path = ctx.metadata_dir / FILENAME
        assert not path.exists()

        # Compute the expected mode the SAME race-free way save() does.
        current = os.umask(0)
        os.umask(current)
        expected_mode = 0o666 & ~current

        save_annotations(ctx, {"1": ["a"]})

        assert path.exists()
        assert stat.S_IMODE(os.stat(path).st_mode) == expected_mode

    def test_save_creates_missing_parent_dir(self, tmp_config):
        """Behaviour preserved from annotator.save: parent dir is created
        if absent (the route relied on this)."""
        ctx = FilmContext.from_config(tmp_config)
        nested = ctx.metadata_dir / "deep" / "nested"
        ctx2 = FilmContext(
            slug=None,
            raw_path=ctx.raw_path,
            data_dir=ctx.data_dir,
            metadata_dir=nested,
            frames_dir=ctx.frames_dir,
            embeddings_dir=ctx.embeddings_dir,
        )
        assert not nested.exists()
        save_annotations(ctx2, {"1": ["x"]})
        assert (nested / FILENAME).exists()
        assert load_annotations(ctx2) == {"1": ["x"]}


# ── build_scene_list ──────────────────────────────────────────────────────────

class TestBuildSceneList:
    def test_empty_dir_yields_empty(self, tmp_config):
        ctx = FilmContext.from_config(tmp_config)
        scenes, desc, ann = build_scene_list(ctx, "no_llm")
        assert scenes == []
        assert desc == {}
        assert ann == {}

    def test_all_filter_returns_every_scene(self, tmp_config, seed_metadata):
        seed_metadata()  # both seeded scenes have valid descriptions
        ctx = FilmContext.from_config(tmp_config)
        scenes, _, _ = build_scene_list(ctx, "all")
        assert {s["scene_id"] for s in scenes} == {351, 352}

    def test_no_llm_filter_drops_scenes_with_valid_description(
        self, tmp_config, seed_metadata
    ):
        # 351 has a valid description, 352 does NOT (omitted from desc).
        seed_metadata(
            descriptions=[
                {"scene_id": 351, "description": "a real description"}
            ]
        )
        ctx = FilmContext.from_config(tmp_config)
        scenes, _, _ = build_scene_list(ctx, "no_llm")
        assert [s["scene_id"] for s in scenes] == [352]

    def test_broken_llm_placeholder_does_not_count_as_valid(
        self, tmp_config, seed_metadata
    ):
        seed_metadata(
            descriptions=[
                {"scene_id": 351, "description": _BROKEN_LLM},
                {"scene_id": 352, "description": "ok"},
            ]
        )
        ctx = FilmContext.from_config(tmp_config)
        scenes, _, _ = build_scene_list(ctx, "no_llm")
        # 351's description is the broken placeholder → still "no llm".
        assert [s["scene_id"] for s in scenes] == [351]

    def test_error_record_does_not_count_as_valid(
        self, tmp_config, seed_metadata
    ):
        seed_metadata(
            descriptions=[
                {"scene_id": 351, "description": "x", "error": "boom"},
                {"scene_id": 352, "description": "ok"},
            ]
        )
        ctx = FilmContext.from_config(tmp_config)
        scenes, _, _ = build_scene_list(ctx, "no_llm")
        assert [s["scene_id"] for s in scenes] == [351]


# ── scene_context / build_scene_panel ─────────────────────────────────────────

class TestSceneContext:
    def test_empty_scene_list_returns_empty_shape(self, tmp_config):
        ctx = FilmContext.from_config(tmp_config)
        out = scene_context(ctx, [], None, {}, {})
        assert out == {
            "scene": None,
            "scene_list": [],
            "total": 0,
            "annotated_count": 0,
        }

    def test_defaults_to_first_scene_when_id_none(
        self, tmp_config, seed_metadata
    ):
        seed_metadata()
        ctx = FilmContext.from_config(tmp_config)
        scenes, desc, ann = build_scene_list(ctx, "all")
        out = scene_context(ctx, scenes, None, desc, ann)
        assert out["scene_id"] == 351
        assert out["current_idx"] == 0
        assert out["prev_id"] is None
        assert out["next_id"] == 352
        assert out["total"] == 2

    def test_defaults_to_first_when_id_not_in_list(
        self, tmp_config, seed_metadata
    ):
        seed_metadata()
        ctx = FilmContext.from_config(tmp_config)
        scenes, desc, ann = build_scene_list(ctx, "all")
        out = scene_context(ctx, scenes, 99999, desc, ann)
        assert out["scene_id"] == 351

    def test_jump_to_specific_scene_sets_nav_edges(
        self, tmp_config, seed_metadata
    ):
        seed_metadata()
        ctx = FilmContext.from_config(tmp_config)
        scenes, desc, ann = build_scene_list(ctx, "all")
        out = scene_context(ctx, scenes, 352, desc, ann)
        assert out["scene_id"] == 352
        assert out["current_idx"] == 1
        assert out["prev_id"] == 351
        assert out["next_id"] is None

    def test_existing_tags_and_annotated_count(
        self, tmp_config, seed_metadata
    ):
        # Default seed: manual annotation only on "352".
        seed_metadata()
        ctx = FilmContext.from_config(tmp_config)
        scenes, desc, ann = build_scene_list(ctx, "all")
        out = scene_context(ctx, scenes, 352, desc, ann)
        assert out["existing_tags"] == ["manual-only", "noite"]
        assert out["tags_value"] == "manual-only, noite"
        # annotated_count counts scenes whose STR id is an annotations
        # key — only "352" here (semantics preserved, NOT changed).
        assert out["annotated_count"] == 1

    def test_llm_shown_only_when_valid(self, tmp_config, seed_metadata):
        seed_metadata(
            descriptions=[
                {"scene_id": 351, "description": "good text"},
                {"scene_id": 352, "description": _BROKEN_LLM},
            ]
        )
        ctx = FilmContext.from_config(tmp_config)
        scenes, desc, ann = build_scene_list(ctx, "all")
        good = scene_context(ctx, scenes, 351, desc, ann)
        broken = scene_context(ctx, scenes, 352, desc, ann)
        assert good["llm"] is not None
        assert broken["llm"] is None

    def test_build_scene_panel_matches_manual_composition(
        self, tmp_config, seed_metadata
    ):
        seed_metadata()
        ctx = FilmContext.from_config(tmp_config)
        scenes, desc, ann = build_scene_list(ctx, "all")
        manual = scene_context(ctx, scenes, 352, desc, ann)
        via_panel = build_scene_panel(ctx, 352, "all")
        assert via_panel == manual


# ── build_annotate_context ────────────────────────────────────────────────────

class TestBuildAnnotateContext:
    def test_no_data_branch(self, tmp_config):
        ctx = FilmContext.from_config(tmp_config)
        out = build_annotate_context(ctx)
        assert out["no_data"] is True
        assert out["all_done"] is False
        assert out["scene"] is None
        assert out["filter"] == "no_llm"

    def test_all_done_when_every_scene_has_valid_llm(
        self, tmp_config, seed_metadata
    ):
        # Both seeded scenes have valid descriptions → no_llm filter empties
        # the list while data exists → all_done is flagged, but the service
        # falls back to filter="all" so the annotation UI stays usable.
        seed_metadata()
        ctx = FilmContext.from_config(tmp_config)
        out = build_annotate_context(ctx, "no_llm")
        assert out["no_data"] is False
        assert out["all_done"] is True
        # Fell back to "all" — scene panel is populated, not blocked
        assert out["filter"] == "all"
        assert out["scene"] is not None
        assert out["total"] == 2

    def test_populated_panel_with_all_filter(
        self, tmp_config, seed_metadata
    ):
        seed_metadata()
        ctx = FilmContext.from_config(tmp_config)
        out = build_annotate_context(ctx, "all")
        assert out["no_data"] is False
        assert out["all_done"] is False
        assert out["scene"]["scene_id"] == 351
        assert out["total"] == 2
        assert out["filter"] == "all"

    def test_scene_id_argument_jumps(self, tmp_config, seed_metadata):
        seed_metadata()
        ctx = FilmContext.from_config(tmp_config)
        out = build_annotate_context(ctx, "all", 352)
        assert out["scene_id"] == 352
        assert out["current_idx"] == 1


# ── save_description ──────────────────────────────────────────────────────────

class TestSaveDescription:
    def test_updates_existing_entry(self, tmp_config, seed_metadata):
        seed_metadata()
        ctx = FilmContext.from_config(tmp_config)
        save_description(ctx, 351, "manually edited text")

        import json
        records = json.loads((ctx.metadata_dir / "scene_descriptions.json").read_text())
        entry = next(r for r in records if r["scene_id"] == 351)
        assert entry["description"] == "manually edited text"

    def test_preserves_other_fields(self, tmp_config, seed_metadata):
        seed_metadata(
            descriptions=[
                {"scene_id": 351, "description": "original", "objects": ["tree"]},
                {"scene_id": 352, "description": "other"},
            ]
        )
        ctx = FilmContext.from_config(tmp_config)
        save_description(ctx, 351, "updated")

        import json
        records = json.loads((ctx.metadata_dir / "scene_descriptions.json").read_text())
        entry = next(r for r in records if r["scene_id"] == 351)
        assert entry["description"] == "updated"
        assert entry["objects"] == ["tree"]

    def test_appends_new_entry_when_scene_not_found(self, tmp_config, seed_metadata):
        seed_metadata(descriptions=[{"scene_id": 351, "description": "exists"}])
        ctx = FilmContext.from_config(tmp_config)
        save_description(ctx, 999, "brand new")

        import json
        records = json.loads((ctx.metadata_dir / "scene_descriptions.json").read_text())
        ids = {r["scene_id"] for r in records}
        assert 999 in ids
        entry = next(r for r in records if r["scene_id"] == 999)
        assert entry["description"] == "brand new"

    def test_creates_file_when_none_exists(self, tmp_config):
        ctx = FilmContext.from_config(tmp_config)
        assert not (ctx.metadata_dir / "scene_descriptions.json").exists()
        save_description(ctx, 351, "first description")

        import json
        records = json.loads((ctx.metadata_dir / "scene_descriptions.json").read_text())
        assert records == [{"scene_id": 351, "description": "first description"}]

    def test_leaves_no_temp_file(self, tmp_config, seed_metadata):
        seed_metadata()
        ctx = FilmContext.from_config(tmp_config)
        save_description(ctx, 351, "clean")

        leftovers = list(ctx.metadata_dir.glob(".scene_descriptions.*.tmp"))
        assert leftovers == []

    def test_overwrite_does_not_duplicate_entry(self, tmp_config, seed_metadata):
        seed_metadata()
        ctx = FilmContext.from_config(tmp_config)
        save_description(ctx, 351, "first edit")
        save_description(ctx, 351, "second edit")

        import json
        records = json.loads((ctx.metadata_dir / "scene_descriptions.json").read_text())
        entries = [r for r in records if r["scene_id"] == 351]
        assert len(entries) == 1
        assert entries[0]["description"] == "second edit"

    def test_scene_panel_reflects_update(self, tmp_config, seed_metadata):
        seed_metadata()
        ctx = FilmContext.from_config(tmp_config)
        save_description(ctx, 351, "rewritten by user")
        panel = build_scene_panel(ctx, 351, "all")
        assert panel["llm"]["description"] == "rewritten by user"
