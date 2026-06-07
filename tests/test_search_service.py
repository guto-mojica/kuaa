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
import pandas as pd
import pytest

from api.services.search import (
    IndexStatus,
    UploadRejected,
    clear_index_cache,
    load_index,
    results_to_dicts,
    validate_upload,
)
from cinemateca.library import FilmContext


@pytest.fixture(autouse=True)
def _isolate_cache():
    """Every test starts and ends with an empty index cache."""
    clear_index_cache()
    yield
    clear_index_cache()


@pytest.fixture()
def clipfree(monkeypatch):
    """Patch OpenClipEmbedder so construction is CLIP-free; .load stays real."""
    import cinemateca.models.clip.openclip as oc

    real_load = oc.OpenClipEmbedder.load

    class _PatchedEmbedder:
        def __init__(self, *a, **k):
            pass

        load = staticmethod(real_load)

        def encode_text(self, query):
            return np.ones(4, dtype="float32")

    monkeypatch.setattr(oc, "OpenClipEmbedder", _PatchedEmbedder)
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

    def test_declared_total_vectors_mismatch_is_corrupt(self, tmp_config, clipfree):
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

    def test_regenerated_index_picked_up_without_restart(self, tmp_config, clipfree):
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
        return pd.DataFrame(
            [
                {
                    "rank": 1,
                    "scene_id": scene_id,
                    "filepath": filepath,
                    "similarity": similarity,
                }
            ]
        )

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


# ── Degenerate-tag display filter ─────────────────────────────────────────────


class TestDegenerateTagFilter:
    """``_filter_degenerate_tags`` drops raw model-output strings that
    leak into ``scene_tags.json`` (full captions, stuck-token repetitions,
    numeric-only, sentence fragments) so the search-tab pill grid stays
    legible without rewriting the underlying tag_index."""

    def _kept(self, tags):
        from api.services.search import _filter_degenerate_tags

        return _filter_degenerate_tags(tags)

    def test_keeps_short_curated_tags(self):
        good = ["dia", "exterior", "interior", "man", "woman", "tree", "sky"]
        assert self._kept(good) == good

    def test_keeps_trailing_period_tags(self):
        # ``rural-field.`` / ``farm.`` are corpus-frequent in jeca_tatu and
        # carry signal; only mid-string ``.`` indicates a sentence fragment.
        assert self._kept(["farm.", "rural-field."]) == ["farm.", "rural-field."]

    def test_drops_full_caption_tags(self):
        long = "a-rural-field-with-a-wooden-fence-and-a-person-riding-a-horse."
        assert self._kept([long, "dia"]) == ["dia"]

    def test_drops_repeated_token_tags(self):
        # 3+ consecutive identical tokens — Moondream stuck-token output.
        bad = [
            "fence-gate-gate-gate-gate-gate-gate",
            "gate-gate-gate",
            "fence-gate-gate-gate",
        ]
        assert self._kept(bad) == []

    def test_drops_pure_digit_tags(self):
        assert self._kept(["1", "42", "man"]) == ["man"]

    def test_drops_sentence_fragments_with_internal_period(self):
        # The ``.`` is mid-string, not trailing → sentence fragment.
        bad = ["the-setting-is-a-farm.-with-cows", "dia.exterior"]
        assert self._kept(bad) == []

    def test_drops_article_led_period_terminated_tags(self):
        # Trailing ``.`` is OK on bare nouns (``farm.``); paired with an
        # article prefix it signals a caption fragment leak.
        bad = [
            "a-baby-in-a-basket.",
            "a-rural-field.",
            "a-tree.",
            "the-setting.",
        ]
        assert self._kept(bad) == []

    def test_drops_long_but_otherwise_innocent_tags(self):
        # 41 chars → over the 40-char threshold even without obvious garbage.
        too_long = "a" * 41
        assert self._kept([too_long, "ok"]) == ["ok"]

    def test_drops_digit_led_enumeration_prefix(self):
        # Moondream's "N-<thing>" listing pattern.
        bad = ["1-cow", "2-buildings", "3-sky", "1-man-wearing-hat"]
        assert self._kept(bad) == []

    def test_drops_de_dup_numeric_suffix(self):
        # "-<digit>" suffix marks Moondream's per-image dedupe.
        bad = ["man-in-hat-2", "woman-in-dress-3", "man-on-horse-5"]
        assert self._kept(bad) == []
        # Sanity: the unsuffixed forms survive.
        assert self._kept(["man-in-hat", "woman-in-dress"]) == [
            "man-in-hat",
            "woman-in-dress",
        ]

    def test_drops_excess_hyphens(self):
        # Curated tags have 0-2 hyphens; >2 = multi-clause sentence garbage.
        bad = ["man-in-plaid-shirt", "person-walking-in-woods"]
        assert self._kept(bad) == []

    def test_empty_input_returns_empty(self):
        assert self._kept([]) == []


# ── search_text similarity floor ──────────────────────────────────────────────


class TestSearchTextMinSimilarity:
    """``search_text(..., min_similarity=X)`` drops result rows whose
    cosine score is below the threshold. CLIP returns top_k unconditionally,
    so without this filter, unrelated queries surface noise scenes
    ('airplane' in a 1959 rural film) at score ~0.22–0.25.
    """

    def _index(self, vectors: list[list[float]]):
        """Build a SearchIndex over L2-normalised ``vectors``."""
        from api.services.search import IndexStatus, SearchIndex

        arr = np.array(vectors, dtype="float32")
        arr /= np.linalg.norm(arr, axis=1, keepdims=True)
        kf_df = pd.DataFrame(
            [{"scene_id": i, "filepath": f"frames/s{i}.jpg"} for i in range(len(vectors))]
        )

        class _Embedder:
            def encode_text(self, q):
                # Query vector: [1, 0] — cosine matches each row's first comp.
                return np.array([1.0, 0.0], dtype="float32")

        return SearchIndex(
            status=IndexStatus.OK,
            embeddings=arr,
            kf_df=kf_df,
            embedder=_Embedder(),
        )

    def test_no_floor_returns_all_top_k(self):
        from api.services.search import search_text

        index = self._index([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        df = search_text(index, "x", tags=[], tag_index={}, top_k=8)
        assert len(df) == 3
        assert df["similarity"].iloc[0] >= df["similarity"].iloc[-1]

    def test_floor_filters_low_scores(self):
        from api.services.search import search_text

        index = self._index([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        # [1,0]→1.0, [0.5,0.5]→0.707, [0,1]→0.0. Floor 0.8 keeps only the
        # perfect match.
        df = search_text(index, "x", tags=[], tag_index={}, top_k=8, min_similarity=0.8)
        assert len(df) == 1
        assert df["scene_id"].iloc[0] == 0
        assert float(df["similarity"].iloc[0]) >= 0.8

    def test_floor_above_everything_returns_empty_df(self):
        from api.services.search import search_text

        index = self._index([[1.0, 0.0], [0.0, 1.0]])
        df = search_text(index, "x", tags=[], tag_index={}, top_k=8, min_similarity=2.0)
        assert df.empty

    def test_floor_zero_is_a_noop(self):
        """``min_similarity=0.0`` is the back-compat default — no rows dropped."""
        from api.services.search import search_text

        index = self._index([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        df = search_text(index, "x", tags=[], tag_index={}, top_k=8, min_similarity=0.0)
        assert len(df) == 3


class TestSceneDedup:
    """``search_text`` and ``aggregate_search`` collapse multi-keyframe-
    per-scene results to one entry per scene, keeping the highest-score
    keyframe. This is the Phase-1 density fix's downstream contract: the
    UI still receives scene-shaped results even though the index now has
    N rows per scene.
    """

    def _multi_kf_index(self):
        """3 scenes × 2 keyframes each = 6 vectors.

        Vector layout (rows L2-normalised after build):
            row 0: scene 1, kf 1 — vec=[1, 0]      similarity to [1,0] = 1.0
            row 1: scene 1, kf 2 — vec=[0.6, 0.8]                       = 0.6
            row 2: scene 2, kf 1 — vec=[0.8, 0.6]                       = 0.8
            row 3: scene 2, kf 2 — vec=[0.4, 0.9]                       ~ 0.41
            row 4: scene 3, kf 1 — vec=[0.3, 1.0]                       ~ 0.29
            row 5: scene 3, kf 2 — vec=[0.1, 1.0]                       ~ 0.10
        Expected dedup order: scene 1 (1.0) > scene 2 (0.8) > scene 3 (0.29).
        And specifically: kf 1 wins both for scene 1 and scene 2.
        """
        from api.services.search import IndexStatus, SearchIndex

        vectors = [
            [1.0, 0.0],
            [0.6, 0.8],
            [0.8, 0.6],
            [0.4, 0.9],
            [0.3, 1.0],
            [0.1, 1.0],
        ]
        arr = np.array(vectors, dtype="float32")
        arr /= np.linalg.norm(arr, axis=1, keepdims=True)
        kf_df = pd.DataFrame(
            [
                {"scene_id": 1, "keyframe_id": "scene_0001_kf_01", "filepath": "s1_k1.jpg"},
                {"scene_id": 1, "keyframe_id": "scene_0001_kf_02", "filepath": "s1_k2.jpg"},
                {"scene_id": 2, "keyframe_id": "scene_0002_kf_01", "filepath": "s2_k1.jpg"},
                {"scene_id": 2, "keyframe_id": "scene_0002_kf_02", "filepath": "s2_k2.jpg"},
                {"scene_id": 3, "keyframe_id": "scene_0003_kf_01", "filepath": "s3_k1.jpg"},
                {"scene_id": 3, "keyframe_id": "scene_0003_kf_02", "filepath": "s3_k2.jpg"},
            ]
        )

        class _Embedder:
            def encode_text(self, q):
                return np.array([1.0, 0.0], dtype="float32")

        return SearchIndex(
            status=IndexStatus.OK,
            embeddings=arr,
            kf_df=kf_df,
            embedder=_Embedder(),
        )

    def test_search_text_dedupes_by_scene_id(self):
        """3 scenes × 2 keyframes → at most 3 result rows, one per scene."""
        from api.services.search import search_text

        index = self._multi_kf_index()
        df = search_text(index, "x", tags=[], tag_index={}, top_k=8)
        assert len(df) == 3, f"expected 3 deduped rows (one per scene), got {len(df)}\n{df}"
        scene_ids = df["scene_id"].tolist()
        assert sorted(scene_ids) == [1, 2, 3]
        # Best-matching keyframe per scene wins (kf 1 for both 1 and 2).
        # SemanticSearch.by_text emits only scene_id/filepath/similarity,
        # so we verify the winning keyframe via filepath.
        fp_by_scene = dict(zip(df["scene_id"], df["filepath"]))
        assert fp_by_scene[1] == "s1_k1.jpg"
        assert fp_by_scene[2] == "s2_k1.jpg"

    def test_search_text_top_k_trims_after_dedup(self):
        """``top_k=2`` returns 2 *scenes*, not 2 keyframes from the same scene."""
        from api.services.search import search_text

        index = self._multi_kf_index()
        df = search_text(index, "x", tags=[], tag_index={}, top_k=2)
        assert len(df) == 2
        assert df["scene_id"].tolist() == [1, 2]

    def test_search_text_dedup_with_min_similarity(self):
        """Dedup composes with the min_similarity floor; the floor is
        applied first (to the raw keyframes), then dedup."""
        from api.services.search import search_text

        index = self._multi_kf_index()
        # Floor 0.5: keeps scene1/kf1 (1.0), scene1/kf2 (0.6), scene2/kf1 (0.8).
        # After dedup: scenes 1 and 2 survive.
        df = search_text(index, "x", tags=[], tag_index={}, top_k=8, min_similarity=0.5)
        assert len(df) == 2
        assert sorted(df["scene_id"].tolist()) == [1, 2]

    def test_search_image_dedupes_by_scene_id(self):
        """Image search follows the same dedup contract as text search."""
        from api.services.search import search_image

        index = self._multi_kf_index()

        # Patch the searcher's by_image to use the same vector logic.
        # (The real by_image needs a JPEG on disk; we stub it out.)
        class _Searcher:
            def __init__(self, embeddings, kf_df, embedder):
                self.embeddings = embeddings
                self.kf_df = kf_df

            def by_image(self, image_path, top_k):
                qv = np.array([1.0, 0.0], dtype="float32")
                scores = self.embeddings @ qv
                order = np.argsort(-scores)
                rows = self.kf_df.iloc[order[:top_k]].copy()
                rows["similarity"] = scores[order[:top_k]]
                return rows.reset_index(drop=True)

        # Monkeypatch SemanticSearch at its post-T9 home in
        # ``cinemateca.search.clip``. The verb hoisted the
        # ``from cinemateca.embeddings import SemanticSearch`` to module
        # scope, so patching the source module ``cinemateca.embeddings``
        # no longer reaches the already-bound name inside ``clip``;
        # patch the module where the verb actually looks up the symbol.
        import cinemateca.search.clip as _clip_mod

        _orig = _clip_mod.SemanticSearch
        _clip_mod.SemanticSearch = _Searcher
        try:
            from pathlib import Path

            df = search_image(index, Path("/tmp/fake.jpg"), top_k=8)
        finally:
            _clip_mod.SemanticSearch = _orig

        assert len(df) == 3, f"expected 3 deduped rows, got {len(df)}"
        assert sorted(df["scene_id"].tolist()) == [1, 2, 3]


class TestAggregateSearchDedup:
    """``aggregate_search`` dedupes by (film_slug, scene_id) post-merge.

    With multiple keyframes per scene the same scene can rank multiple
    times before the global sort; the dedup keeps only the best-scoring
    keyframe per scene so each result card maps to one scene.
    """

    def test_aggregate_dedupes_by_film_scene(self, tmp_path, monkeypatch):
        """Two films × multiple keyframes-per-scene → at most one hit
        per (film_slug, scene_id) in the final result."""
        import sys
        from types import SimpleNamespace

        import cinemateca.search.aggregate as _csa_ref  # noqa: F401 — ensure loaded
        from cinemateca.library import register_film

        csa = sys.modules["cinemateca.search.aggregate"]

        # ── Two-film library with 3 keyframes per scene each ──
        library_dir = tmp_path / "library"
        for slug, title in (("film_a", "Film A"), ("film_b", "Film B")):
            register_film(
                library_dir, slug=slug, title=title, year=None, raw_filename=f"{slug}.mp4"
            )
            film_dir = library_dir / slug
            (film_dir / "metadata").mkdir(parents=True, exist_ok=True)
            (film_dir / "embeddings").mkdir(parents=True, exist_ok=True)
            # 2 scenes × 3 keyframes each — metadata 1:N rows.
            kf_meta = []
            for scene_id in (1, 2):
                for kf_pos in (1, 2, 3):
                    kf_meta.append(
                        {
                            "scene_id": scene_id,
                            "keyframe_id": f"scene_{scene_id:04d}_kf_{kf_pos:02d}",
                            "filepath": f"library/{slug}/frames/{scene_id}_{kf_pos}.jpg",
                            "start_time_s": float(scene_id * 10),
                            "end_time_s": float(scene_id * 10 + 5),
                        }
                    )
            (film_dir / "metadata" / "keyframes_metadata.json").write_text(json.dumps(kf_meta))

        cfg = SimpleNamespace(
            paths=SimpleNamespace(library_dir=str(library_dir)),
            embeddings=SimpleNamespace(
                filename="keyframe_embeddings.npy",
                mapping_filename="index_mapping.json",
            ),
        )

        # ── Stub _get_search_index to return a 6-vector index per film ──
        from api.services.search import IndexStatus, SearchIndex

        # Each film has 6 rows: 2 scenes × 3 keyframes. Scene 1's first
        # keyframe scores highest in both films.
        vectors = np.array(
            [
                [1.0, 0.0],
                [0.6, 0.8],
                [0.4, 0.9],  # scene 1 (kf 1/2/3)
                [0.8, 0.6],
                [0.3, 1.0],
                [0.1, 1.0],  # scene 2 (kf 1/2/3)
            ],
            dtype="float32",
        )
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        def fake_index(_cfg, slug):
            kf_df = pd.DataFrame(
                [
                    {"scene_id": 1, "filepath": f"{slug}_s1_kf1.jpg"},
                    {"scene_id": 1, "filepath": f"{slug}_s1_kf2.jpg"},
                    {"scene_id": 1, "filepath": f"{slug}_s1_kf3.jpg"},
                    {"scene_id": 2, "filepath": f"{slug}_s2_kf1.jpg"},
                    {"scene_id": 2, "filepath": f"{slug}_s2_kf2.jpg"},
                    {"scene_id": 2, "filepath": f"{slug}_s2_kf3.jpg"},
                ]
            )
            return SearchIndex(
                status=IndexStatus.OK,
                embeddings=vectors,
                kf_df=kf_df,
                embedder=None,
            )

        class _Embedder:
            def encode_text(self, q):
                return np.array([1.0, 0.0], dtype="float32")

        monkeypatch.setattr(csa, "_get_search_index", fake_index)
        monkeypatch.setattr(csa, "_get_embedder", lambda _cfg: _Embedder())

        # ── Run aggregate search ──
        hits = csa.aggregate_search(cfg, query="x", modality="text", top_k=10)

        # 2 films × 2 scenes = 4 dedup'd hits max.
        assert len(hits) == 4, f"expected 4 deduped hits, got {len(hits)}\n{hits}"
        keys = {(h["film_slug"], h["scene_id"]) for h in hits}
        assert keys == {("film_a", 1), ("film_a", 2), ("film_b", 1), ("film_b", 2)}

    def test_aggregate_dedup_picks_best_keyframe_per_scene(self, tmp_path, monkeypatch):
        """The kept keyframe per (film, scene) is the one with the
        highest cosine score in that scene."""
        import sys
        from types import SimpleNamespace

        import cinemateca.search.aggregate as _csa_ref  # noqa: F401 — ensure loaded
        from api.services.search import IndexStatus, SearchIndex
        from cinemateca.library import register_film

        csa = sys.modules["cinemateca.search.aggregate"]

        library_dir = tmp_path / "library"
        register_film(
            library_dir, slug="film_a", title="Film A", year=None, raw_filename="film_a.mp4"
        )
        film_dir = library_dir / "film_a"
        (film_dir / "metadata").mkdir(parents=True, exist_ok=True)
        (film_dir / "embeddings").mkdir(parents=True, exist_ok=True)
        (film_dir / "metadata" / "keyframes_metadata.json").write_text(
            json.dumps(
                [
                    {"scene_id": 1, "filepath": "f1.jpg", "start_time_s": 0.0},
                ]
            )
        )

        cfg = SimpleNamespace(
            paths=SimpleNamespace(library_dir=str(library_dir)),
            embeddings=SimpleNamespace(
                filename="keyframe_embeddings.npy",
                mapping_filename="index_mapping.json",
            ),
        )

        # One scene, 3 keyframes; only the second one has the best score.
        vectors = np.array(
            [
                [0.6, 0.8],  # kf 1 → cos 0.6
                [1.0, 0.0],  # kf 2 → cos 1.0 (winner)
                [0.4, 0.9],  # kf 3 → cos 0.4
            ],
            dtype="float32",
        )
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        def fake_index(_cfg, slug):
            kf_df = pd.DataFrame(
                [
                    {"scene_id": 1, "filepath": "kf_01.jpg"},
                    {"scene_id": 1, "filepath": "kf_02.jpg"},  # winner
                    {"scene_id": 1, "filepath": "kf_03.jpg"},
                ]
            )
            return SearchIndex(
                status=IndexStatus.OK,
                embeddings=vectors,
                kf_df=kf_df,
                embedder=None,
            )

        class _Embedder:
            def encode_text(self, q):
                return np.array([1.0, 0.0], dtype="float32")

        monkeypatch.setattr(csa, "_get_search_index", fake_index)
        monkeypatch.setattr(csa, "_get_embedder", lambda _cfg: _Embedder())

        hits = csa.aggregate_search(cfg, query="x", modality="text", top_k=10)
        assert len(hits) == 1
        assert hits[0]["keyframe_path"] == "kf_02.jpg", (
            f"expected best-matching keyframe (kf_02), got {hits[0]['keyframe_path']}"
        )
