# Offline and privacy notes

Status: public documentation draft.

KUAA is designed as a local-first tool for private visual collections. This
document explains what that means today, where network access can still
happen, and which claims should not be made.

## Short version

After dependencies and model weights are installed, the application is intended
to process videos locally:

- source videos stay on the local machine,
- extracted frames stay on the local machine,
- generated metadata stays on the local machine,
- embeddings stay on the local machine,
- annotations stay on the local machine,
- search queries are handled by the local app.

The app should be described as offline-first or local-first, not as a formally
certified privacy or compliance system.

## What stays local

The following artifacts are stored under local project paths configured in
`config/default.yaml` or `config/local.yaml`:

- raw videos under `data/raw`,
- sampled frames under `data/frames/sample`,
- scene keyframes under `data/frames/scenes/keyframes_content`,
- metadata under `data/metadata`,
- embeddings under `data/embeddings`,
- manual annotations under `data/metadata/manual_annotations.json`,
- logs under `logs` when file logging is enabled.

Search is local:

- text queries are embedded by the local image/text embedding backend,
- image queries are stored temporarily by the web request path and embedded
  locally,
- ranking uses NumPy dot products over local embeddings.

The UI is local:

- FastAPI serves HTML from local Jinja templates,
- HTMX and the SSE extension are vendored under `web/static/js`,
- fonts are vendored under `web/static/fonts`,
- CSS is local,
- keyframes are served from the local `data` directory through the app's
  `/media` mount.

## When network access may happen

Network access may happen during setup or first use:

- Python packages are installed from package indexes unless already cached.
- SigLIP2/OpenCLIP model weights may download on first model load.
- YOLOv8 weights may download on first object-detector load.
- MTCNN/facenet-pytorch weights may download on first face-detector load.
- Moondream 2 weights may download through Hugging Face on first scene
  description load.
- GGUF files may download through `huggingface-hub` when the GGUF backend is
  selected.

After packages and weights are available locally, the intended runtime path does
not require sending collection data to external APIs.

## What is not sent to APIs by design

The current model stack is local/keyless. The project does not require sending
these to a hosted AI API:

- videos,
- keyframes,
- generated descriptions,
- generated tags,
- embeddings,
- manual annotations,
- text search queries,
- image search uploads.

If future contributors add hosted models or telemetry, that must be documented
and disabled by default for the archive/offline use case.

## Sensitive-data cautions

Even local processing can create sensitive derived data:

- keyframes may expose people, places, documents, or culturally sensitive scenes;
- embeddings can reveal similarity relationships inside a collection;
- generated descriptions may include inaccurate or biased interpretations;
- manual annotations may encode curatorial decisions or private research notes;
- logs may include file paths and error messages.

Treat derived artifacts as collection data. Do not publish them unless the
source material and metadata are approved for public release.

## Face detection policy

The current face backend detects faces and counts them. It does not identify
people by name and should not be described as face recognition.

Recommended public wording:

> The system detects visible faces for scene-level metadata. It does not perform
> identity recognition.

For sensitive collections, face detection should be configurable and can be
disabled.

## Public demo guidance

The public demo should avoid private or institutionally sensitive footage.

Recommended demo rules:

- use public-domain or clearly permissive footage,
- document source and rights status,
- do not include private annotations,
- do not include internal file paths in screenshots,
- do not commit model caches or large downloaded weights,
- use precomputed demo artifacts only when redistribution rights are clear.

## Offline verification checklist

Before making offline claims publicly, verify them directly:

- Install dependencies and model weights.
- Start the app locally.
- Enable browser offline mode in devtools.
- Hard reload the web app.
- Confirm HTML, CSS, JavaScript, icons, fonts, and keyframes load from local
  routes.
- Confirm no requests go to external CDNs.
- Run search against an existing local index.
- Confirm processing does not use hosted AI APIs.

## Claims to avoid

Avoid these claims:

- "Compliant with privacy law."
- "Certified secure."
- "Never uses the network."
- "No risk of sensitive data leakage."
- "The AI metadata is accurate."
- "Safe for all commercial use."

Safer claims:

- "Runs locally after installation and model download."
- "Does not require hosted AI APIs for processing or search."
- "Designed for private visual collections."
- "Supports human review and correction."
- "Keeps generated artifacts on disk under local project paths."
