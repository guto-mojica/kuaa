"""
tests/test_search_service.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 3c: direct unit tests for the extracted search service
(``api/services/search.py``).

These are NEW units (the service did not exist before Phase 3c), so
they ADD coverage on top of the Phase 0/1/2 route regression net and
the 3a/3b service units — they do not replace them. They pin the
service's public surface directly (no HTTP round-trip):

  * mtime/size-aware cache invalidation: a regenerated index is picked
    up WITHOUT a restart / manual clear (the prior @lru_cache bug);
  * shape validation: missing -> MISSING; row mismatch -> CORRUPT;
    declared total_vectors mismatch -> CORRUPT; well-formed -> OK
    (never raises IndexError out of the service);
  * upload guards: oversize / wrong content-type / bad suffix rejected,
    valid inputs return the right suffix;
  * FilmContext path usage (the index is read under the ctx's
    embeddings_dir, not an inline cfg.paths read).

Hermetic: built on the shared ``tmp_config`` factory fixture
(no GPU, no CLIP, no real video, no repo ``data/`` access — enforced by
tmp_config's path guard). ``CLIPEmbedder`` is patched in the embeddings
module so the constructed embedder is CLIP-free while ``.load`` keeps
its real (no-validation) behaviour — the same seam Phase-2 used.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import pandas as pd

from api.services.film_context import FilmContext
from api.services.search import (
    IndexStatus,
    UploadRejected,
    clear_index_cache,
    load_index,
    results_to_dicts,
    validate_upload,
)


@pytest.fixture(autouse=True)
def _isolate_cache():
    """Every test starts and ends with an empty index cache."""
    clear_index_cache()
    yield
    clear_index_cache()


@pytest.fixture()
def clipfree(monkeypatch):
    """Patch CLIPEmbedder so construction is CLIP-free; .load stays real."""
    import cinemateca.embeddings as emb

    real_load = emb.CLIPEmbedder.load

    class _PatchedEmbedder:
        def __init__(self, *a, **k):
            pass

        load = staticmethod(real_load)

        def encode_text(self, query):
            return np.ones(4, dtype="float32")

    monkeypatch.setattr(emb, "CLIPEmbedder", _PatchedEmbedder)
    return _PatchedEmbedder


def _write_index(cfg, n_emb: int, scene_ids: list, *, total_vectors=None):
    """Write an embeddings .npy (n_emb rows) + mapping (len(scene_ids))."""
    emb_dir = Path(cfg.paths.embeddings_dir)
    np.save(emb_dir / cfg.embeddings.filename, np.eye(4, dtype="float32")[:n_emb])
    mapping = {
        "model": "stub",
        "dimension": 4,
        "total_vectors": len(scene_ids) if total_vectors is None else total_vectors,
        "normalized": True,
        "keyframe_paths": [f"frames/s{s}.jpg" for s in scene_ids],
        "scene_ids": scene_ids,
    }
    (emb_dir / cfg.embeddings.mapping_filename).write_text(json.dumps(mapping))


def _load(cfg):
    ctx = FilmContext.from_config(cfg)
    return load_index(
        ctx,
        mapping_filename=cfg.embeddings.mapping_filename,
        embeddings_filename=cfg.embeddings.filename,
    )


# ── Shape validation ──────────────────────────────────────────────────────────

class TestLoadIndexValidation:
    def test_missing_index_is_missing_not_raise(self, tmp_config):
        idx = _load(tmp_config)
        assert idx.status is IndexStatus.MISSING
        assert not idx.ok
        assert idx.embeddings is None

    def test_wellformed_index_is_ok(self, tmp_config, clipfree):
        _write_index(tmp_config, 2, [1, 2])
        idx = _load(tmp_config)
        assert idx.status is IndexStatus.OK
        assert idx.ok
        assert idx.embeddings.shape[0] == 2
        assert len(idx.kf_df) == 2
        assert idx.embedder is not None

    def test_row_mismatch_is_corrupt_not_indexerror(self, tmp_config, clipfree):
        # 3 embedding rows, 2 mapping rows — the Phase-2 crash case.
        _write_index(tmp_config, 3, [1, 2])
        idx = _load(tmp_config)
        assert idx.status is IndexStatus.CORRUPT
        assert not idx.ok
        assert "mismatch" in idx.detail

    def test_declared_total_vectors_mismatch_is_corrupt(
        self, tmp_config, clipfree
    ):
        # rows are self-consistent (2 vs 2) but the mapping lies about
        # total_vectors — still incoherent, must not be served.
        _write_index(tmp_config, 2, [1, 2], total_vectors=99)
        idx = _load(tmp_config)
        assert idx.status is IndexStatus.CORRUPT

    def test_unreadable_mapping_is_corrupt(self, tmp_config):
        emb_dir = Path(tmp_config.paths.embeddings_dir)
        np.save(emb_dir / tmp_config.embeddings.filename, np.eye(2, dtype="float32"))
        (emb_dir / tmp_config.embeddings.mapping_filename).write_text("{not json")
        idx = _load(tmp_config)
        assert idx.status is IndexStatus.CORRUPT


# ── mtime/size cache invalidation ─────────────────────────────────────────────

class TestCacheInvalidation:
    def test_second_load_is_cached_same_object(self, tmp_config, clipfree):
        _write_index(tmp_config, 2, [1, 2])
        a = _load(tmp_config)
        b = _load(tmp_config)
        assert a is b  # served from cache, no reload

    def test_regenerated_index_picked_up_without_restart(
        self, tmp_config, clipfree
    ):
        """The core fix: mutate the on-disk index (different size) and
        the very next load reflects it — no process restart, no manual
        cache_clear (the old @lru_cache-by-path never did this)."""
        _write_index(tmp_config, 2, [1, 2])
        first = _load(tmp_config)
        assert len(first.kf_df) == 2

        # Regenerate with a different shape -> different (mtime,size).
        _write_index(tmp_config, 3, [10, 20, 30])
        second = _load(tmp_config)
        assert second is not first
        assert len(second.kf_df) == 3
        assert list(second.kf_df["scene_id"]) == [10, 20, 30]

    def test_corrupt_then_fixed_is_picked_up(self, tmp_config, clipfree):
        """A corrupt index that is later regenerated correctly must
        transition CORRUPT -> OK without a restart."""
        _write_index(tmp_config, 3, [1, 2])  # corrupt (row mismatch)
        assert _load(tmp_config).status is IndexStatus.CORRUPT

        _write_index(tmp_config, 2, [1, 2])  # fixed
        good = _load(tmp_config)
        assert good.status is IndexStatus.OK
        assert good.ok

    def test_index_disappearing_invalidates_cache(self, tmp_config, clipfree):
        _write_index(tmp_config, 2, [1, 2])
        assert _load(tmp_config).ok

        emb_dir = Path(tmp_config.paths.embeddings_dir)
        (emb_dir / tmp_config.embeddings.filename).unlink()
        (emb_dir / tmp_config.embeddings.mapping_filename).unlink()
        assert _load(tmp_config).status is IndexStatus.MISSING


# ── FilmContext path usage ────────────────────────────────────────────────────

class TestFilmContextWiring:
    def test_load_reads_under_ctx_embeddings_dir(self, tmp_config, clipfree):
        """The service resolves the index under ``ctx.embeddings_dir``
        (FilmContext), not an inline cfg.paths read — proven by writing
        the index ONLY under that dir and getting an OK load."""
        ctx = FilmContext.from_config(tmp_config)
        assert ctx.embeddings_dir == Path(tmp_config.paths.embeddings_dir)
        _write_index(tmp_config, 1, [7])
        idx = load_index(
            ctx,
            mapping_filename=tmp_config.embeddings.mapping_filename,
            embeddings_filename=tmp_config.embeddings.filename,
        )
        assert idx.ok
        assert list(idx.kf_df["scene_id"]) == [7]


# ── Upload guards ─────────────────────────────────────────────────────────────

class TestValidateUpload:
    def test_empty_rejected(self):
        with pytest.raises(UploadRejected):
            validate_upload("a.jpg", "image/jpeg", b"")

    def test_oversize_rejected(self):
        from api.services.search import MAX_UPLOAD_BYTES

        with pytest.raises(UploadRejected):
            validate_upload("a.jpg", "image/jpeg", b"x" * (MAX_UPLOAD_BYTES + 1))

    def test_non_image_content_type_rejected(self):
        with pytest.raises(UploadRejected):
            validate_upload("a.jpg", "application/pdf", b"data")

    def test_bad_suffix_rejected(self):
        with pytest.raises(UploadRejected):
            validate_upload("payload.exe", "image/jpeg", b"data")

    def test_missing_suffix_and_no_ctype_rejected(self):
        with pytest.raises(UploadRejected):
            validate_upload("noext", None, b"data")

    def test_valid_jpeg_returns_suffix(self):
        assert validate_upload("photo.JPG", "image/jpeg", b"data") == ".jpg"

    def test_valid_png_returns_suffix(self):
        assert validate_upload("x.png", "image/png", b"data") == ".png"

    def test_no_suffix_but_image_ctype_defaults_jpg(self):
        assert validate_upload("blob", "image/png", b"data") == ".jpg"

    def test_content_type_with_charset_param_ok(self):
        assert validate_upload("x.png", "image/png; charset=binary", b"d") == ".png"


# ── results_to_dicts timecode enrichment ─────────────────────────────────────

class TestResultsToDicts:
    def _df(self, scene_id: int, filepath: str, similarity: float = 0.9):
        return pd.DataFrame([{
            "rank": 1, "scene_id": scene_id,
            "filepath": filepath, "similarity": similarity,
        }])

    def test_no_meta_no_timecode_key(self, tmp_path):
        df = self._df(1, str(tmp_path / "frames" / "s1.jpg"))
        rows = results_to_dicts(df, tmp_path)
        assert "timecode" not in rows[0]

    def test_with_meta_adds_smpte_timecode(self, tmp_path):
        (tmp_path / "frames").mkdir()
        kf = tmp_path / "frames" / "s1.jpg"
        kf.touch()
        df = self._df(1, str(kf))
        meta_by_scene = {1: {"scene_id": 1, "start_time_s": 83.0, "start_frame": 1992}}
        rows = results_to_dicts(df, tmp_path, meta_by_scene, fps=24.0)
        assert rows[0]["timecode"] == "00:01:23:00"

    def test_with_meta_zero_start_time_empty_timecode(self, tmp_path):
        df = self._df(1, str(tmp_path / "s1.jpg"))
        meta_by_scene = {1: {"scene_id": 1, "start_time_s": 0.0, "start_frame": 0}}
        rows = results_to_dicts(df, tmp_path, meta_by_scene, fps=24.0)
        assert rows[0]["timecode"] == ""

    def test_with_meta_missing_scene_no_timecode(self, tmp_path):
        df = self._df(99, str(tmp_path / "s99.jpg"))
        meta_by_scene = {1: {"scene_id": 1, "start_time_s": 10.0}}
        rows = results_to_dicts(df, tmp_path, meta_by_scene, fps=24.0)
        assert "timecode" not in rows[0]

    def test_img_url_still_resolved(self, tmp_path):
        (tmp_path / "frames").mkdir()
        kf = tmp_path / "frames" / "s1.jpg"
        kf.touch()
        df = self._df(1, str(kf))
        rows = results_to_dicts(df, tmp_path)
        assert rows[0]["img_url"] == "/media/frames/s1.jpg"
