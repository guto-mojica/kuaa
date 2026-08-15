"""Guards against the 2026-08-14 MPS-OOM incident.

A VLM whose device is failing does not raise — it emits a repetition loop
that passes every text guard and gets indexed as scene content. These tests
pin the detector, the record-level rejection, the token cap that keeps a
runaway generation cheap, the memory release, and the circuit breaker.

Model fully mocked; never loads weights, never downloads.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kuaa.errors import (
    ModelError,
    is_degenerate_response,
    is_error_response,
    is_unusable_response,
)
from kuaa.models.describer._common import build_metadata, degenerate_fields

# Verbatim from logs/kuaa.log, chronopolis_1982 scene 2 (truncated).
REAL_DEGENERATE = "s. s. s. s. s. s. s. s. s. s. s. s. s. s. s. s. s. s. s s. s. s."
# Verbatim from the same run, scene 1 — healthy.
REAL_HEALTHY = "A silhouetted figure climbs a tall building, gripping a rope for support."


# ── detector ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        REAL_DEGENERATE,
        "s s s s s s s s s s s s",  # same failure without punctuation
        "the the the the the the the the the",
        "a b a b a b a b a b a b a b a b a b a b a b a b",  # long low-ratio loop
    ],
)
def test_degenerate_texts_are_caught(text):
    assert is_degenerate_response(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "outdoor",
        "day",
        "2 people talking",
        "tree, fence, wooden barrel",
        "rural field",
        REAL_HEALTHY,
        # Terse answers must never trip the guard, however repetitive.
        "no no no",
        # A legitimately long caption with normal vocabulary.
        "Two men in trench coats stand near a large electrical cable reel, "
        "one of them holding a hat while the other looks away toward the street.",
    ],
)
def test_healthy_texts_are_not_caught(text):
    assert not is_degenerate_response(text)


def test_degenerate_text_is_not_error_shaped():
    """The whole point: the old guard could not see this."""
    assert not is_error_response(REAL_DEGENERATE)
    assert is_unusable_response(REAL_DEGENERATE)


def test_unusable_still_catches_captured_failures():
    assert is_unusable_response("Error: command buffer exited with error status.")


# ── record-level rejection ──────────────────────────────────────────────────


def _row(scene_id: int = 7) -> pd.Series:
    return pd.Series({"filepath": "kf.jpg", "scene_id": scene_id})


def _healthy_raw() -> dict:
    return {
        "description": REAL_HEALTHY,
        "location": "outdoor",
        "setting": "urban street",
        "time_of_day": "day",
        "people_and_action": "2 people talking",
        "objects": "tree, fence",
    }


def test_healthy_raw_builds_a_good_row():
    meta = build_metadata(_row(), _healthy_raw())
    assert "error" not in meta
    assert meta["description"] == REAL_HEALTHY
    assert meta["tags"]


def test_any_degenerate_field_makes_the_whole_row_an_error():
    """One pinned field means the decoder was broken — trust none of it."""
    raw = _healthy_raw()
    raw["setting"] = REAL_DEGENERATE
    meta = build_metadata(_row(), raw)
    assert "error" in meta
    assert "setting" in meta["error"]
    assert meta["tags"] == []
    assert "description" not in meta
    # Provenance survives so the row is debuggable and resumable.
    assert meta["scene_id"] == 7
    assert meta["_raw_responses"]["setting"] == REAL_DEGENERATE


def test_degenerate_setting_never_reaches_the_tag_vocabulary():
    """Regression: setting is kebab-cased straight into tags."""
    raw = _healthy_raw()
    raw["setting"] = REAL_DEGENERATE
    meta = build_metadata(_row(), raw)
    assert not any("s-s-s" in t for t in meta["tags"])


def test_degenerate_fields_reports_every_offender():
    raw = _healthy_raw()
    raw["description"] = REAL_DEGENERATE
    raw["objects"] = REAL_DEGENERATE
    assert degenerate_fields(raw) == ["description", "objects"]


def test_degenerate_rows_are_reprocessed_on_resume():
    """Error rows are dropped by describe_batch, so the scene gets another go."""
    raw = _healthy_raw()
    raw["description"] = REAL_DEGENERATE
    meta = build_metadata(_row(scene_id=3), raw)
    processed_ids = {r["scene_id"] for r in [meta] if "error" not in r}
    assert 3 not in processed_ids


# ── token cap ───────────────────────────────────────────────────────────────


class _RecordingModel:
    """Stands in for the 2025-01-09 remote code: query() honours the cap."""

    def __init__(self):
        self.query_settings: list[dict] = []
        self.answer_question_calls = 0

    def encode_image(self, _img):
        return object()

    def query(self, _enc, _prompt, settings=None):
        self.query_settings.append(settings or {})
        return {"answer": "a plausible answer"}

    def answer_question(self, _enc, _prompt, _tok, max_new_tokens=None):
        self.answer_question_calls += 1
        return "a plausible answer"


class _LegacyModel(_RecordingModel):
    """Stands in for a revision that predates query()."""

    query = None  # type: ignore[assignment]


def _stub_pil(monkeypatch):
    import PIL.Image as _PILImage

    class _StubImg:
        def convert(self, _m):
            return self

        def resize(self, _s, _r):
            return self

    monkeypatch.setattr(_PILImage, "open", lambda _p: _StubImg())


def _backend_with(monkeypatch, model):
    from kuaa.models.describer import transformers_hf

    def _fake_load(self):
        self._model = model
        self._tokenizer = object()

    monkeypatch.setattr(transformers_hf.MoondreamTransformersDescriber, "_load_model", _fake_load)
    _stub_pil(monkeypatch)
    return transformers_hf.MoondreamTransformersDescriber()


def test_token_cap_is_passed_through_query(monkeypatch):
    """answer_question silently ignores max_new_tokens; query does not."""
    from kuaa.models.describer._common import PROMPTS

    model = _RecordingModel()
    backend = _backend_with(monkeypatch, model)
    backend.describe("kf.jpg")

    assert model.answer_question_calls == 0, "must not use the uncapped path"
    caps = [s["max_tokens"] for s in model.query_settings]
    assert caps == [mx for _prompt, mx in PROMPTS.values()]


def test_legacy_revision_falls_back_loudly(monkeypatch, caplog):
    model = _LegacyModel()
    backend = _backend_with(monkeypatch, model)
    with caplog.at_level("WARNING"):
        backend.describe("kf.jpg")
    assert model.answer_question_calls > 0
    warnings = [r for r in caplog.records if "IGNORA o teto de tokens" in r.message]
    assert len(warnings) == 1, "the fallback must warn exactly once, not per prompt"


# ── memory release ──────────────────────────────────────────────────────────


def test_describe_batch_releases_the_model(monkeypatch):
    backend = _backend_with(monkeypatch, _RecordingModel())
    df = pd.DataFrame([{"filepath": "a.jpg", "scene_id": 1}])
    backend.describe_batch(df)
    assert backend._model is None
    assert backend._enc_cache is None


def test_release_is_idempotent_on_a_never_loaded_backend():
    from kuaa.models.describer import transformers_hf

    backend = transformers_hf.MoondreamTransformersDescriber()
    backend.release()
    backend.release()
    assert backend._model is None


def test_encode_drops_the_previous_frame_before_allocating(monkeypatch):
    """A held EncodedImage owns a 384 MiB KV cache; two must never overlap."""
    backend = _backend_with(monkeypatch, _RecordingModel())
    backend._load_model()

    seen: list[object] = []

    def _tracking_encode(_img):
        seen.append(backend._enc_cache)
        return object()

    backend._model.encode_image = _tracking_encode
    backend._encode("a.jpg")
    backend._encode("b.jpg")
    assert seen == [None, None], "cache must be cleared before the new allocation"


# ── circuit breaker ─────────────────────────────────────────────────────────


class _DegenerateModel(_RecordingModel):
    """A backend that has stopped working, exactly as MPS OOM presents."""

    def __init__(self, healthy_first: int = 0):
        super().__init__()
        self.healthy_first = healthy_first
        self.scenes_seen = 0

    def encode_image(self, _img):
        self.scenes_seen += 1
        return object()

    def query(self, _enc, _prompt, settings=None):
        if self.scenes_seen <= self.healthy_first:
            return {"answer": REAL_HEALTHY}
        return {"answer": REAL_DEGENERATE}


def _df(n: int) -> pd.DataFrame:
    return pd.DataFrame([{"filepath": f"{i}.jpg", "scene_id": i} for i in range(1, n + 1)])


def test_breaker_aborts_after_three_consecutive_degenerate_scenes(monkeypatch):
    backend = _backend_with(monkeypatch, _DegenerateModel())
    with pytest.raises(ModelError, match="3 cenas consecutivas"):
        backend.describe_batch(_df(410))


def test_breaker_does_not_fire_on_an_isolated_bad_scene(monkeypatch):
    """One odd frame is not a broken backend."""
    model = _DegenerateModel(healthy_first=0)

    # Degenerate only on scene 2 of 4.
    def _query(_enc, _prompt, settings=None):
        return {"answer": REAL_DEGENERATE if model.scenes_seen == 2 else REAL_HEALTHY}

    model.query = _query  # type: ignore[method-assign]
    backend = _backend_with(monkeypatch, model)
    out = backend.describe_batch(_df(4))
    assert len(out) == 4
    assert sum("error" in r for r in out) == 1


def test_breaker_checkpoints_the_good_rows_before_aborting(monkeypatch, tmp_path):
    """The 20 hours of the incident must not be lost a second time."""
    import json

    backend = _backend_with(monkeypatch, _DegenerateModel(healthy_first=2))
    ckpt = tmp_path / "scene_descriptions.json"
    with pytest.raises(ModelError):
        backend.describe_batch(_df(50), checkpoint_path=ckpt)

    saved = json.loads(ckpt.read_text())
    good = [r for r in saved if "error" not in r]
    assert len(good) == 2, "healthy scenes must survive the abort"
    assert {r["scene_id"] for r in good} == {1, 2}


def test_breaker_releases_the_model_on_the_abort_path(monkeypatch):
    backend = _backend_with(monkeypatch, _DegenerateModel())
    with pytest.raises(ModelError):
        backend.describe_batch(_df(10))
    assert backend._model is None


def test_breaker_can_be_disabled(monkeypatch):
    backend = _backend_with(monkeypatch, _DegenerateModel())
    backend.max_consecutive_degenerate = 0
    out = backend.describe_batch(_df(5))
    assert len(out) == 5
    assert all("error" in r for r in out)


def test_abort_message_names_the_likely_cause(monkeypatch):
    backend = _backend_with(monkeypatch, _DegenerateModel())
    with pytest.raises(ModelError) as exc:
        backend.describe_batch(_df(10))
    assert "MPS" in str(exc.value)
    assert "Restart the process" in str(exc.value)


# ── checkpoint coherence ────────────────────────────────────────────────────
#
# The tag index is derived from the descriptions. If a checkpoint writes one
# without the other, a resumed-then-consumed run indexes a generation that no
# longer exists — and the BM25 tags surface is built straight from that file,
# so nothing downstream can detect the skew. Both backends used to get this
# wrong in different ways: gguf wrote tags on the periodic checkpoint but not
# on the abort path, transformers_hf never wrote them at all.


def _read_checkpoint_pair(ckpt, tags):
    import json

    return json.loads(ckpt.read_text()), json.loads(tags.read_text())


def test_periodic_checkpoint_writes_the_tag_index(monkeypatch, tmp_path):
    backend = _backend_with(monkeypatch, _RecordingModel())
    backend.checkpoint_interval = 2
    ckpt = tmp_path / "scene_descriptions.json"
    tags = tmp_path / backend.tags_filename

    backend.describe_batch(_df(4), checkpoint_path=ckpt)

    assert tags.exists(), "tag index must be written beside every checkpoint"
    saved, tag_index = _read_checkpoint_pair(ckpt, tags)
    tagged = {sid for sids in tag_index.values() for sid in sids}
    assert tagged == {str(r["scene_id"]) for r in saved if "error" not in r}


def test_abort_checkpoint_writes_the_tag_index(monkeypatch, tmp_path):
    """The abort path is a checkpoint too — it must not skip the derived file."""
    backend = _backend_with(monkeypatch, _DegenerateModel(healthy_first=2))
    ckpt = tmp_path / "scene_descriptions.json"
    tags = tmp_path / backend.tags_filename

    with pytest.raises(ModelError):
        backend.describe_batch(_df(50), checkpoint_path=ckpt)

    assert tags.exists(), "aborting must not leave descriptions without their index"
    saved, tag_index = _read_checkpoint_pair(ckpt, tags)
    good = {str(r["scene_id"]) for r in saved if "error" not in r}
    tagged = {sid for sids in tag_index.values() for sid in sids}
    assert tagged == good, "index and descriptions must be the same generation"


def test_tag_index_never_outlives_a_stale_generation(monkeypatch, tmp_path):
    """A pre-existing index from an older run is overwritten, not left behind."""
    import json

    ckpt = tmp_path / "scene_descriptions.json"
    tags = tmp_path / "scene_tags.json"
    tags.write_text(json.dumps({"ghost-tag": ["999"]}), encoding="utf-8")

    backend = _backend_with(monkeypatch, _RecordingModel())
    backend.checkpoint_interval = 1
    backend.describe_batch(_df(2), checkpoint_path=ckpt)

    tag_index = json.loads(tags.read_text())
    assert "ghost-tag" not in tag_index
    assert "999" not in {sid for sids in tag_index.values() for sid in sids}
