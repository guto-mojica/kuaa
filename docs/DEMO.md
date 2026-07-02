# Reproducible Demo

This phase provides a public, populated demo path that does not require private
archive data and does not require a reviewer to run the full ML pipeline before
seeing the app.

Primary source: Library of Congress item `00694220`, *The Great Train Robbery*
(1903). Provenance and rights notes are in `docs/DEMO_DATA.md`.

## Five-Minute Quickstart

```bash
uv sync --extra full --group dev
uv run python scripts/prepare_demo.py --download
KUAA_CONFIG=config/demo.yaml uv run kuaa serve
```

Open `http://127.0.0.1:8501`.

What this path does:

- Downloads the versioned demo artifact bundle from the configured release URL.
- Extracts it under `data/demo/runtime/`.
- Downloads the public source video when a direct LOC video URL can be
  discovered, or tells you where to place it manually.
- Validates keyframes, metadata JSON, embeddings, index mapping, and optional
  checksums before printing the run command.

If the release artifact is hosted elsewhere, override the bundle URL:

```bash
uv run python scripts/prepare_demo.py --download \
  --bundle-url https://example.org/kuaa-demo-v1.zip
```

Validate local artifacts without network:

```bash
uv run python scripts/prepare_demo.py --check
```

Measure the prepared demo index:

```bash
uv run python scripts/run_eval.py \
  --config config/demo.yaml \
  --queries data/eval/archive_demo_queries.yaml \
  --output-dir data/eval/reports
```

Evaluation outputs are generated under `data/eval/reports/` and are ignored by
git by default. See `docs/EVALUATION.md` for metric definitions.

## Runtime Layout

The demo config points all local data paths at `data/demo/runtime/`:

```text
data/demo/runtime/
├── raw/
├── frames/scenes/keyframes_content/
├── metadata/
│   ├── keyframes_metadata.json
│   ├── scene_descriptions.json
│   ├── scene_tags.json
│   ├── visual_analysis.json
│   └── manual_annotations.json        optional
└── embeddings/
    ├── keyframe_embeddings.npy
    └── index_mapping.json
```

The release ZIP may contain either this full path or the layout rooted at
`metadata/`, `frames/`, and `embeddings/`; `scripts/prepare_demo.py` supports
both.

## What Works From Precomputed Artifacts

The populated Scenes and Annotate tabs work from the artifact bundle alone.
The Search tab can load the precomputed index; running a new text or image query
may still need CLIP model weights on first use. After dependencies, model
weights, source video, and artifacts are downloaded, the web UI uses local files
and vendored static assets.

## Full Processing Path

The demo also keeps a real processing path for users who want to run the
pipeline:

```bash
uv run kuaa process \
  data/demo/runtime/raw/the-great-train-robbery-1903.mp4 \
  --config config/demo.yaml
```

`config/demo.yaml` is CPU-conservative and limits the LLM pass for faster
experimentation. It does not overwrite `config/local.yaml`.

## Expected Review Flow

1. Run `scripts/prepare_demo.py --download`.
2. Start `KUAA_CONFIG=config/demo.yaml uv run kuaa serve`.
3. Open Scenes and verify real keyframes and descriptions are visible.
4. Open Annotate and verify seeded/generated metadata can be reviewed.
5. Run a search query after CLIP weights are available.
6. Open Processing and verify the demo source video appears when downloaded.

## Two-minute walkthrough

Target length: about two minutes. Use the app started via
`KUAA_CONFIG=config/demo.yaml uv run kuaa serve`.

### Opening

"KUAA is an offline multimodal workbench for turning archival video into
searchable, human-reviewable scene metadata. This demo uses a public Library of
Congress film, not private institutional data."

### Scenes

Open Scenes.

"The pipeline has already segmented the film into scenes, extracted keyframes,
and attached generated visual metadata. A reviewer can scan the film visually
without opening a video editor."

Show:

- keyframe grid
- timecodes
- generated descriptions
- tags and environment labels

### Search

Open Search.

Example queries:

- `train station`
- `men with horses`
- `interior rail car`

"Search is semantic, so the user can ask for visual concepts instead of exact
keywords. The index is local; search queries and keyframes do not go to a hosted
AI API."

### Annotate

Open Annotate.

"The machine output is a first pass, not a final catalog. Curators can correct
or add tags scene by scene. Manual annotations are merged back into search."

Show:

- next/previous scene navigation
- existing tags
- adding one correction tag
- save feedback

### Processing

Open Processing.

"The demo opens quickly from precomputed artifacts, but the full processing path
is still available. The demo config is isolated from local user config and uses
CPU-conservative settings."

Show the selected source video if present.

### Close

"The important product point is not just a movie search demo. It is a local,
inspectable workflow for private visual collections: generate useful metadata,
let humans correct it, and keep the artifacts traceable."
