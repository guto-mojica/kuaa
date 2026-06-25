"""T12: structured-logging wiring — json_logs toggle + access-log field contract.

Verifies:
- ``setup_logging(cfg)`` with ``json_logs=True`` installs ``_JsonFormatter``
  on root handlers and each emitted line parses as valid JSON with the
  expected keys (ts / level / name / msg / request_id).
- The ``api.access`` logger emits a ``LogRecord`` carrying the F5 extra
  fields (method / path / status / duration_ms / request_id) on every
  request processed by ``RequestContextMiddleware``.

No new dependencies — stdlib ``logging`` + the existing F5 formatter.
"""

from __future__ import annotations

import json
import logging

import pytest

pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg_json_logs(tmp_path):
    """Return a Settings with json_logs=True and file logging disabled."""
    from kuaa.config import load_config

    cfg = load_config(project_root=tmp_path, ensure_dirs=False)
    cfg.logging.json_logs = True
    cfg.logging.to_file = False
    # Rebase paths to tmp so any residual dir-creation calls are safe.
    for attr in (
        "data_dir",
        "raw_dir",
        "frames_dir",
        "metadata_dir",
        "embeddings_dir",
        "library_dir",
        "models_dir",
        "outputs_dir",
        "logs_dir",
    ):
        (tmp_path / attr).mkdir(parents=True, exist_ok=True)
        setattr(cfg.paths, attr, tmp_path / attr)
    return cfg


# ---------------------------------------------------------------------------
# Test 1: json_logs toggle wires _JsonFormatter; output is valid JSON
# ---------------------------------------------------------------------------


def test_json_logs_toggle_installs_formatter_and_emits_valid_json(tmp_path):
    """With json_logs=True, setup_logging installs _JsonFormatter on root
    handlers; formatting a record through that formatter yields valid JSON
    with keys ts / level / name / msg / request_id."""
    from kuaa.config.loader import _JsonFormatter, setup_logging

    cfg = _make_cfg_json_logs(tmp_path)

    # setup_logging calls logging.basicConfig(force=True) which replaces all
    # existing root handlers.  Run it first so the installed handlers are the
    # ones we inspect.
    setup_logging(cfg)

    # 1. At least one root handler must carry _JsonFormatter.
    root = logging.getLogger()
    json_handlers = [h for h in root.handlers if isinstance(h.formatter, _JsonFormatter)]
    assert json_handlers, (
        "setup_logging(json_logs=True) did not install _JsonFormatter on any handler"
    )

    # 2. Formatting a synthetic record through _JsonFormatter yields valid JSON
    #    with every required key.
    formatter = json_handlers[0].formatter
    assert isinstance(formatter, _JsonFormatter)

    record = logging.LogRecord(
        name="kuaa.test_t12",
        level=logging.INFO,
        pathname="test",
        lineno=0,
        msg="hello structured world",
        args=(),
        exc_info=None,
    )
    # Without request_id extra — request_id should be null.
    line = formatter.format(record)
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        pytest.fail(f"_JsonFormatter produced non-JSON output: {exc!r}\nLine: {line!r}")

    for key in ("ts", "level", "name", "msg", "request_id"):
        assert key in obj, f"JSON log missing key {key!r}: {obj}"

    assert obj["request_id"] is None, (
        f"request_id should be null when not set via extra; got {obj['request_id']!r}"
    )
    assert obj["msg"] == "hello structured world"
    assert obj["level"] == "INFO"

    # 3. With request_id in extra — request_id must be populated.
    record_with_rid = logging.LogRecord(
        name="api.access",
        level=logging.INFO,
        pathname="middleware",
        lineno=0,
        msg="GET /health -> 200",
        args=(),
        exc_info=None,
    )
    record_with_rid.request_id = "test-rid-1234"
    line_rid = formatter.format(record_with_rid)
    obj_rid = json.loads(line_rid)
    assert obj_rid["request_id"] == "test-rid-1234", (
        f"request_id not propagated; got {obj_rid['request_id']!r}"
    )


# ---------------------------------------------------------------------------
# Test 2: access-log record carries F5 extra fields
# ---------------------------------------------------------------------------


def test_access_log_record_carries_f5_fields(client, caplog):
    """RequestContextMiddleware emits an api.access record with all F5 fields.

    F5 contract: method / path / status / duration_ms / request_id are
    present as attributes on the LogRecord (set via ``extra=``).
    """
    with caplog.at_level(logging.INFO, logger="api.access"):
        r = client.get("/health")

    assert r.status_code == 200, (
        f"/health returned {r.status_code}; F5 access-log test requires a 200 route"
    )

    access_records = [rec for rec in caplog.records if rec.name == "api.access"]
    assert access_records, (
        "No api.access log record found; RequestContextMiddleware must be registered"
    )

    rec = access_records[-1]
    for field in ("method", "path", "status", "duration_ms", "request_id"):
        assert hasattr(rec, field), (
            f"api.access LogRecord is missing field {field!r}; "
            f"available attrs: {[a for a in vars(rec) if not a.startswith('_')]}"
        )

    # Sanity-check values
    assert rec.method == "GET"
    assert rec.path == "/health"
    assert rec.status == 200
    assert isinstance(rec.duration_ms, float)
    assert rec.request_id  # non-empty string


# ---------------------------------------------------------------------------
# Test 3: json_logs=False (default) does NOT install _JsonFormatter
# ---------------------------------------------------------------------------


def test_json_logs_false_does_not_install_json_formatter(tmp_path):
    """Regression guard: default json_logs=False leaves plain text formatter."""
    from kuaa.config import load_config
    from kuaa.config.loader import _JsonFormatter, setup_logging

    cfg = load_config(project_root=tmp_path, ensure_dirs=False)
    cfg.logging.json_logs = False
    cfg.logging.to_file = False

    setup_logging(cfg)

    json_handlers = [
        h for h in logging.getLogger().handlers if isinstance(h.formatter, _JsonFormatter)
    ]
    assert not json_handlers, "setup_logging(json_logs=False) should NOT install _JsonFormatter"
