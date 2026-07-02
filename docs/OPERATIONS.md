# Operations notes

Status: current operating model

This project is an offline, single-user applied AI workbench. These notes
describe the current operating model for local runs, exports, run manifests,
and failure behavior.

## Operating model

- Run locally with `uv run kuaa serve` or `uv run kuaa serve --config config/demo.yaml`.
- The app serves local assets and generated keyframes from the configured
  `paths.data_dir`.
- The app uses `data/library/films.json` plus per-film artifact directories
  under `data/library/<slug>/...`. Legacy flat artifact paths still exist for
  older configs and compatibility code paths.
- The Processing tab runs one active job at a time. A second start request is
  rejected with `409`.
- Terminal processing jobs are retained in memory only for recent UI history.

## Generated artifacts

Default paths come from `config/default.yaml` or the selected config override.

| Artifact | Default path | Producer |
|---|---|---|
| Film registry | `data/library/films.json` | Library add/register |
| Video properties | `data/library/<slug>/metadata/video_properties.json` | Frame extraction |
| Keyframe metadata | `data/library/<slug>/metadata/keyframes_metadata.json` | Scene detection |
| Keyframe images | `data/library/<slug>/frames/scenes/keyframes_content/` | Scene detection |
| Visual analysis | `data/library/<slug>/metadata/visual_analysis.json` | Visual analysis |
| Scene descriptions | `data/library/<slug>/metadata/scene_descriptions.json` | LLM description |
| Scene tags | `data/library/<slug>/metadata/scene_tags.json` | LLM description |
| Manual annotations | `data/library/<slug>/metadata/manual_annotations.json` | Annotation UI |
| Visual embeddings | `data/library/<slug>/embeddings/keyframe_embeddings.npy` | Embeddings |
| Visual index mapping | `data/library/<slug>/embeddings/index_mapping.json` | Embeddings |
| Run manifest | `data/metadata/run_manifest.json` | Pipeline or Processing tab |

## Run manifests

Every `CatalogPipeline.run()` execution writes `run_manifest.json` through the
shared manifest helper. That helper currently writes to the configured flat
metadata directory (`cfg.paths.metadata_dir`), even when the processed film's
primary artifacts live under `data/library/<slug>/...`. The Processing-tab
worker writes the same manifest shape after success, error, blocked downstream
steps, or cancellation.

The manifest records:

- input path, existence, size, and modified time,
- config SHA-256 plus a JSON-compatible config snapshot,
- selected domain pack id, label, and path,
- configured model backends and model revision hints,
- step states, durations, errors, and output paths,
- expected output artifact paths and whether each exists,
- run start/end timestamps and terminal status.

Manifest writing is best effort. A manifest write failure is logged but does not
replace the real pipeline result.

## Structured exports

The app exports the current catalog on request:

```bash
curl -L http://localhost:8501/api/export/catalog.json -o catalog_export.json
curl -L http://localhost:8501/api/export/catalog.csv -o catalog_export.csv
```

Exports are built from current metadata artifacts and the selected domain pack's
`export_mapping`.

JSON exports include:

- export schema version,
- generation timestamp,
- scene count,
- selected domain identity,
- source artifact paths,
- missing optional artifacts,
- domain-shaped scene records.

CSV exports use the same scene records and serialize list/dict values as JSON
strings inside cells.

Required artifact: at least one registered film with
`metadata/keyframes_metadata.json`. If required metadata is missing, export
routes return `404` with a clear message. Other missing artifacts are recorded
in `missing_artifacts`.

## Failure behavior

Processing steps are dependency-aware in the web runner:

- `visual_analysis` requires keyframe image files from scene detection.
- `embeddings` and `llm_description` require `keyframes_metadata.json`.
- If a prerequisite step fails or a required input artifact is missing, the
  downstream step is marked `blocked`.
- A job with any `error` or `blocked` step ends with terminal status `error`.
- Cancellation is cooperative and ends with terminal status `cancelled`.
- The SSE stream emits one terminal event (`done`, `error`, or `cancelled`) and
  then closes.

CLI `kuaa process` keeps the historical `pipeline.stop_on_error` behavior for
the full pipeline path, and also writes a run manifest.

## Structured logging and request-ID correlation

### JSON log toggle

By default, logs are human-readable plain text. For machine-parseable,
one-JSON-object-per-line output (suitable for log aggregators), enable the
`json_logs` toggle in `config/local.yaml`:

```yaml
logging:
  json_logs: true
```

Each JSON line carries:

| Key | Type | Description |
|---|---|---|
| `ts` | string | Timestamp (ISO-ish, local time) |
| `level` | string | Log level (`INFO`, `WARNING`, …) |
| `name` | string | Logger name (e.g. `api.access`, `kuaa.search.clip`) |
| `msg` | string | Formatted log message |
| `request_id` | string \| null | Correlation ID — present on access-log lines, `null` elsewhere |

The formatter is `_JsonFormatter` in `src/kuaa/config/loader.py`, installed
by `setup_logging(cfg)` when `cfg.logging.json_logs is True`.

### Request-ID correlation (X-Request-ID)

Every HTTP request handled by the FastAPI app is assigned a unique UUID by
`RequestContextMiddleware` (`api/middleware.py`). The ID is:

- **Echoed** from the inbound `X-Request-ID` header if the client provides one.
- **Generated** as a fresh UUID v4 otherwise.
- **Stamped** on the response as the `X-Request-ID` header.
- **Stored** on `request.state.request_id` for downstream handlers (e.g. SSE
  streams log with the same ID).

One structured access-log line is emitted on the `api.access` logger per
request, carrying these `LogRecord` extra fields:

| Field | Type | Example |
|---|---|---|
| `request_id` | str | `"3fa85f64-…"` |
| `method` | str | `"GET"` |
| `path` | str | `"/api/search"` |
| `status` | int | `200` |
| `duration_ms` | float | `14.3` |

With `json_logs: true`, these fields appear directly in the JSON object so log
aggregators can filter by `request_id`, `path`, or `status` without regex
parsing.

### Enabling JSON logs at runtime

```bash
# Via config override (persistent across restarts):
#   add `logging:\n  json_logs: true` to config/local.yaml

# Eyeball one live JSON access line:
uv run kuaa serve --no-reload
# then in another terminal:
curl -s http://127.0.0.1:8501/health
# the server terminal will print a JSON object for that request
```

### Tests

`tests/test_structured_logging.py` verifies end-to-end:

1. `setup_logging` with `json_logs=True` installs `_JsonFormatter` on root handlers.
2. A formatted record parses as valid JSON with all required keys.
3. `request_id` is populated when set via `extra`; `null` otherwise.
4. The `api.access` `LogRecord` from a real request carries all request-ID fields.

`tests/test_middleware.py` and `tests/test_request_id.py` verify the
`X-Request-ID` header contract independently.

---

## Release gates

Run these before tagging a release:

```bash
uv run pytest -q
uv run ruff check .
uv run mypy src
git diff --check
```

For the public demo path:

```bash
uv run python scripts/prepare_demo.py --check
uv run python scripts/run_eval.py \
  --config config/demo.yaml \
  --queries data/eval/archive_demo_queries.yaml \
  --output-dir data/eval/reports
```

The demo artifact bundle must be present before claiming populated-demo metrics.

## Operational limits

- This is not an authenticated web service. Bind to localhost unless you control
  the network.
- Model first-run downloads can require network access. After dependencies and
  model weights are installed, processing/search are local.
- Generated artifacts can be large. Keep demo runtime artifacts and evaluation
  reports out of git unless they are intentionally small and documented.
- CSV exports are for interchange and review. JSON exports are the richer
  reloadable format.
- The web worker allows one active job at a time even though the library can
  hold multiple films.
- Global prototype controls such as share, notifications, settings, and import
  are not visible in launch chrome.
