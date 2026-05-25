"""build_rimas_context threads lambda_diversity + k_candidates to find_rhymes."""

from __future__ import annotations


def _patch_find_rhymes_capture(monkeypatch):
    """Replace find_rhymes with a stub that captures kwargs."""
    from api.services import rhymes_service as svc

    captured: dict = {}

    def fake_find_rhymes(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(svc, "find_rhymes", fake_find_rhymes)
    return captured


def _patch_anchor_resolution(monkeypatch, tmp_path):
    """Stub _load_scene_meta to return a truthy anchor so find_rhymes is reached."""
    from api.services import rhymes_service as svc

    monkeypatch.setattr(svc, "_load_scene_meta", lambda cfg, slug, sid: {"scene_id": sid})


def _minimal_cfg(tmp_path):
    """Build a config namespace satisfying build_rimas_context's reads."""
    from types import SimpleNamespace

    (tmp_path / "library").mkdir()
    return SimpleNamespace(
        paths=SimpleNamespace(library_dir=str(tmp_path / "library")),
        rimas=SimpleNamespace(top_n=8, mmr_lambda=0.5, threshold=0.75),
        retrieval=SimpleNamespace(rhymes=SimpleNamespace(diversity=0.5, k_candidates=30)),
    )


def test_explicit_lambda_diversity_overrides_cfg(monkeypatch, tmp_path):
    from api.services.rhymes_service import build_rimas_context

    captured = _patch_find_rhymes_capture(monkeypatch)
    _patch_anchor_resolution(monkeypatch, tmp_path)
    cfg = _minimal_cfg(tmp_path)
    build_rimas_context(cfg=cfg, anchor="any/1", lambda_diversity=0.3, k_candidates=25)
    assert captured.get("lambda_diversity") == 0.3
    assert captured.get("k_candidates") == 25


def test_cfg_default_is_threaded_when_explicit_none(monkeypatch, tmp_path):
    from api.services.rhymes_service import build_rimas_context

    captured = _patch_find_rhymes_capture(monkeypatch)
    _patch_anchor_resolution(monkeypatch, tmp_path)
    cfg = _minimal_cfg(tmp_path)
    build_rimas_context(cfg=cfg, anchor="any/1")
    # cfg.retrieval.rhymes.{diversity,k_candidates} → (0.5, 30)
    assert captured.get("lambda_diversity") == 0.5
    assert captured.get("k_candidates") == 30


def test_fallback_when_cfg_lacks_retrieval_rhymes(monkeypatch, tmp_path):
    """A cfg without retrieval.rhymes should fall through to hard defaults
    (0.5 / 30) — keeps the service usable before Task 3.3 lands the config block."""
    from types import SimpleNamespace

    from api.services.rhymes_service import build_rimas_context

    captured = _patch_find_rhymes_capture(monkeypatch)
    _patch_anchor_resolution(monkeypatch, tmp_path)
    (tmp_path / "library").mkdir()
    cfg = SimpleNamespace(
        paths=SimpleNamespace(library_dir=str(tmp_path / "library")),
        rimas=SimpleNamespace(top_n=8, mmr_lambda=0.5, threshold=0.75),
        # No retrieval attr at all
    )
    build_rimas_context(cfg=cfg, anchor="any/1")
    assert captured.get("lambda_diversity") == 0.5
    assert captured.get("k_candidates") == 30


def test_returned_context_surfaces_effective_lambda(monkeypatch, tmp_path):
    """The dict key 'mmr_lambda' should reflect the LIVE lambda, not the
    legacy cfg.rimas.mmr_lambda."""
    from api.services.rhymes_service import build_rimas_context

    _patch_find_rhymes_capture(monkeypatch)
    _patch_anchor_resolution(monkeypatch, tmp_path)
    cfg = _minimal_cfg(tmp_path)
    ctx = build_rimas_context(cfg=cfg, anchor="any/1", lambda_diversity=0.3, k_candidates=25)
    assert ctx["mmr_lambda"] == 0.3
    assert ctx["k_candidates"] == 25
