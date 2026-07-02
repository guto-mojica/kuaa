# API reference

Status: current FastAPI/HTMX surface

The FastAPI app is a local, single-user interface. Most existing `/api/*`
endpoints return HTMX HTML fragments because they are called by the web UI.
Export and eval endpoints return JSON, CSV, or JSON acknowledgements where the
UI needs structured data.

Start the app:

```bash
uv run kuaa serve
# or
uv run kuaa serve --config config/demo.yaml
```

(`uv run app.py` still works — it's a legacy entrypoint kept for back-compat
that delegates to the same server.)

Base URL: `http://localhost:8501`

## Constraints

- The app is designed for local/offline use, not public internet exposure.
- The app supports a registry-backed multi-film library. Omit `?film=<slug>`
  for aggregate views; include it for per-film views.
- The Processing tab allows one active pipeline job at a time.
- Search endpoints require a prepared embeddings index.
- Export endpoints require at least one registered film with
  `metadata/keyframes_metadata.json`.
- `/eval` is disabled unless `EVAL_ADMIN_TOKEN` is set and supplied by query
  parameter or cookie.
- `/api/share/link` is not implemented and no launch UI points at it.

## Full-page routes

| Method | Path | Response | Purpose |
|---|---|---|---|
| `GET` | `/` | HTML | Search page with base app chrome |
| `GET` | `/search` | HTML | Search page |
| `GET` | `/scenes` | HTML | Scene browsing page |
| `GET` | `/annotate` | HTML | Annotation page |
| `GET` | `/processing` | HTML | Processing page |
| `GET` | `/rimas` | HTML | Cross-film visual-rhymes page |
| `GET` | `/about` | HTML | Full-page About fallback |
| `GET` | `/eval` | HTML | Admin-gated eval grading UI |

## HTMX tab fragments

These routes return partial HTML and are intended for in-app tab swaps.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/tab/search` | Search panel |
| `GET` | `/tab/scenes` | Scene browsing panel |
| `GET` | `/tab/annotate` | Annotation panel |
| `GET` | `/tab/processing` | Processing panel |
| `GET` | `/tab/rimas` | Visual-rhymes panel |

## Search

### Text search

```http
GET /api/search?q=people%20outdoors&tags=exterior&top_k=8&retriever=hybrid
```

Response: HTML fragment for `partials/search_results.html`.

Parameters:

| Name | Type | Notes |
|---|---|---|
| `q` | string | Queries shorter than two characters return an empty body |
| `tags` | repeated string | Optional tag filters |
| `film` | string | Optional film slug; omit for aggregate cross-film search |
| `top_k` | integer | Number of results, default `8` unless the UI sends its stored preference |
| `retriever` | string | `hybrid`, `clip`, or `bm25`; text path only |
| `sem_w` | float | Semantic weight for hybrid search |
| `bm25_w` | float | Lexical weight for hybrid search |
| `reranker_enabled` | bool | Accepted/logged for compatibility, not exposed by the current UI and not yet applied by production dispatchers |

Missing or corrupt index behavior: returns the no-index HTML state, not a 500.

### Image search

```http
POST /api/search/image?film=jeca_tatu&top_k=8
Content-Type: multipart/form-data

file=@reference.jpg
```

Response: HTML fragment for search results.

Accepted uploads: JPEG/PNG image uploads that pass the service validation.
Rejected uploads return a rendered error state in the results fragment.

## Scenes

### Filter scene grid

```http
GET /api/scenes?film=jeca_tatu&tags=exterior&q=station&group=film&sort=timecode
```

Response: HTML fragment for `partials/scenes_grid.html`.

Parameters:

| Name | Type | Notes |
|---|---|---|
| `tags` | repeated string | Intersects scene ids across selected tags |
| `q` | string | Case-insensitive description keyword filter |
| `film` | string | Optional film slug; omit for aggregate grid |
| `group` | string | `film`, `tipo`, or `none` |
| `sort` | string | `timecode`, `duration`, or `pins` |
| `bucket` | string | Optional tipo bucket such as `exterior`, `interior`, `cartela`, `dialogo`, or `transicao` |

### Scene inspector

```http
GET /api/scenes/12/inspector?film=jeca_tatu&tab=properties&kind=cenas
```

Response: right-pane HTML fragment. `kind=buscar` renders the Search inspector;
`kind=cenas` renders the Scenes inspector. Unknown scenes return `404`.

## Annotations

### Load one scene panel

```http
GET /api/annotate/scene?id=351&filter=no_llm&film=jeca_tatu&tab=comments
```

Response: HTML fragment for one annotation scene panel.

### Save scene tags

```http
POST /api/annotate/save
Content-Type: application/x-www-form-urlencoded

scene_id=351&filter=all&tab=annotations&tags=railroad, exterior, multiple people
```

Response: updated scene panel HTML.

Tags are normalized to lowercase with spaces converted to hyphens.

### Edit description form

```http
GET /api/annotate/description/edit?scene_id=351&filter=all
```

Response: description edit HTML fragment.

### Save description

```http
POST /api/annotate/description
Content-Type: application/x-www-form-urlencoded

scene_id=351&filter=all&description=A%20train%20station%20scene
```

Response: updated scene panel HTML.

### Clear scene tags

```http
POST /api/annotate/clear
Content-Type: application/x-www-form-urlencoded

scene_id=351&filter=all
```

Response: updated scene panel HTML.

## Visual rhymes

### Rimas tab

```http
GET /tab/rimas?anchor=jeca_tatu/12&lambda=0.5&k_candidates=30
```

Response: full Rimas tab fragment. `anchor` uses `slug/scene_id`. Missing or
unresolvable anchors render an empty state, not a 500.

### Echo grid

```http
GET /api/rimas/echoes?anchor=jeca_tatu/12&lambda=0.5&k_candidates=30
```

Response: echo-grid HTML fragment for `#rimas-echoes`.

### Rimas inspector

```http
GET /api/rimas/inspector?anchor=jeca_tatu/12&echo=edwin_porter-the_great_train_robbery_1903/3
```

Response: right-pane HTML fragment comparing the anchor and selected echo.

## Processing

### Start a pipeline job

```http
POST /api/pipeline/start
Content-Type: application/x-www-form-urlencoded

video_path=/absolute/path/to/video.mp4&steps=scene_detection&steps=embeddings
```

Response: processing job HTML fragment.

If `steps` is omitted, all steps are selected:

- `frame_extraction`
- `scene_detection`
- `visual_analysis`
- `embeddings`
- `llm_description`

If another job is already running, the route returns `409`.

### Stream job progress

```http
GET /api/pipeline/stream/{job_id}
Accept: text/event-stream
```

Events:

| Event | Meaning |
|---|---|
| `update` | Step state changed |
| `done` | Terminal success |
| `error` | Terminal failure or blocked downstream step |
| `cancelled` | Terminal user cancellation |

The stream closes after one terminal event.

### Cancel a job

```http
POST /api/pipeline/cancel/{job_id}
```

Response: updated job HTML fragment.

Cancellation is cooperative. The worker polls between steps and records the
terminal state as `cancelled`.

### Poll one job card

```http
GET /api/pipeline/job-card/{job_id}
```

Response: active-job card HTML fragment. Active cards use this as a polling
fallback around the SSE stepper/log stream.

## Eval

### Page

```http
GET /eval?token=secret
```

Response: standalone eval grading UI. Requires `EVAL_ADMIN_TOKEN=secret` or an
equivalent `eval_admin` cookie.

### Save grade

```http
POST /api/eval/grade?token=secret
Content-Type: application/x-www-form-urlencoded

query_id=Q001&scene_id=jeca_tatu:12&grade=3
```

Response: JSON acknowledgement. Valid grades are `-1`, `0`, `1`, `2`, and `3`.

### Metrics

```http
GET /api/eval/metrics?token=secret&query_id=Q001
```

Response: JSON metrics bundle with `p_at_3`, `p_at_5`, `ndcg_at_5`,
`inversions`, and `histogram`. Omitting `query_id` returns graded query ids.

## Structured exports

### JSON catalog export

```http
GET /api/export/catalog.json
Accept: application/json
```

Response: `application/json` with `Content-Disposition:
attachment; filename="catalog_export.json"`.

Example shape:

```json
{
  "export": {
    "schema_version": "1.0",
    "generated_at": "2026-05-20T16:00:00Z",
    "scene_count": 2,
    "domain": {
      "id": "archive",
      "label": "Film archive",
      "path": "/repo/config/domains/archive.yaml"
    },
    "artifacts": {
      "keyframes_metadata": "/data/metadata/keyframes_metadata.json"
    },
    "missing_artifacts": ["embeddings", "run_manifest"]
  },
  "scenes": [
    {
      "scene_id": 351,
      "keyframe_path": "/data/frames/scenes/keyframes_content/Scene-351.jpg",
      "description": "A man walking outdoors at dawn.",
      "tags": ["dia", "exterior"]
    }
  ]
}
```

The scene fields come from the selected domain pack's `export_mapping`.

Missing `keyframes_metadata.json` returns `404` with a clear error message.
Optional artifacts appear under `missing_artifacts`.

### CSV catalog export

```http
GET /api/export/catalog.csv
Accept: text/csv
```

Response: `text/csv; charset=utf-8` with `Content-Disposition:
attachment; filename="catalog_export.csv"`.

CSV columns are the union of exported scene fields. Lists and dictionaries are
serialized as JSON strings inside cells.

## Library and app utilities

| Method | Path | Response | Purpose |
|---|---|---|---|
| `GET` | `/api/library/filter?q=term` | HTML | Legacy library tree filter |
| `GET` | `/api/library/tree?q=term` | HTML | Left-pane body filter |
| `GET` | `/api/library/select/{slug}` | Redirect header | Navigate to `/scenes?film={slug}` |
| `GET` | `/api/library/add-form` | HTML | Inline add-film form |
| `POST` | `/api/library/add` | HTML | Register a film and refresh the tree |
| `GET` | `/api/library/remove-confirm/{slug}` | HTML | Remove confirmation fragment |
| `POST` | `/api/library/remove/{slug}` | HTML | Deregister a film, optionally wiping data |
| `GET` | `/api/about` | HTML | About modal fragment |
| `GET` | `/api/locale/{code}` | Redirect or HTMX redirect | Set `pt_BR` or `en` locale cookie |
| `GET` | `/api/palette/search?q=term` | JSON | Command-palette grouped results |
| `GET` | `/health`, `/ready` | JSON | Liveness/readiness probes |

## OpenAPI

FastAPI also exposes its generated docs at `/docs` and `/openapi.json` when the
app is running. Because many endpoints intentionally return HTML fragments, this
document is the canonical behavior reference for integrators.
