# Demo Data Directory

This directory tracks only the M1 demo manifest and documentation. Runtime
artifacts are downloaded or generated locally and are intentionally ignored by
git.

Tracked:

- `manifest.json` — source, bundle, expected layout, and optional checksums.
- `README.md` — this note.

Ignored runtime path:

- `data/demo/runtime/`

Prepare the populated demo:

```bash
uv run python scripts/prepare_demo.py --download
uv run app.py --config config/demo.yaml
```

Validate an already prepared demo without network access:

```bash
uv run python scripts/prepare_demo.py --check
```

The primary M1 source is Library of Congress item `00694220`, *The Great Train
Robbery* (1903). See `docs/DEMO_DATA.md` for provenance and rights notes.

