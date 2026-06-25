"""Static project-data builders for the About surface.

These three functions return project-wide constants that never change at
runtime (every install ships the same models and stack). Extracted from
``about_service.py`` to keep that module ≤ 250 LOC; re-imported there
so all call sites are unaffected.
"""

from __future__ import annotations

from typing import Any


def model_attributions() -> list[dict[str, Any]]:
    """Return the model-attribution cards rendered in the About modal.

    Order follows the pipeline: visual embedding (CLIP), scene description
    (Moondream), object detection (YOLO), then face detection (MTCNN). Each
    entry has:

      * ``key``    — one- or two-char badge text (drives the coloured
                     ``.ab-model .ico`` square at the start of the card).
      * ``color``  — colour variant for the badge: ``""`` (accent purple,
                     the default) / ``"yellow"`` / ``"green"`` / ``"orange"``.
                     Maps to the corresponding
                     ``.ab-model .ico.<color>`` rule in ``about.css``.
      * ``name``   — HuggingFace / GitHub identifier, rendered in mono.
      * ``version``— short version tag shown next to the name.
      * ``role``   — short sentence describing what the model does.
      * ``org``    — owning organisation, shown right-aligned.
      * ``lic``    — license string shown in the right-most pill.
      * ``repo_url``— GitHub repo. Optional — when empty/None the "repo"
                     anchor is omitted by the template.
    """
    return [
        {
            "key": "C",
            "color": "",
            "name": "openai/clip-vit-large-patch14",
            "version": "L/14",
            "role": "Visual embedding",
            "org": "OpenAI",
            "lic": "MIT",
            "repo_url": "https://github.com/openai/CLIP",
        },
        {
            "key": "M",
            "color": "yellow",
            "name": "vikhyatk/moondream2",
            "version": "v2",
            "role": "Scene description",
            "org": "Vikhyat",
            "lic": "Apache-2",
            "repo_url": "https://github.com/vikhyat/moondream",
        },
        {
            "key": "Y",
            "color": "green",
            "name": "ultralytics/yolov8m",
            "version": "v8m",
            "role": "Object detection",
            "org": "Ultralytics",
            "lic": "AGPL-3",
            "repo_url": "https://github.com/ultralytics/ultralytics",
        },
        {
            "key": "F",
            "color": "orange",
            "name": "facenet/mtcnn",
            "version": "",
            "role": "Face detection",
            "org": "facenet-pytorch",
            "lic": "MIT",
            "repo_url": "https://github.com/timesler/facenet-pytorch",
        },
    ]


def tech_stack() -> list[dict[str, Any]]:
    """Return the tech-stack pills shown in the Stack section.

    Each entry has a ``label`` (visible mono text) and an optional
    ``kind`` colour variant: ``""`` (default neutral grey), ``"ac"``,
    ``"green"``, ``"yellow"``, ``"pink"``, or ``"orange"`` — mapped to
    ``.ab-stack-pill.<kind>`` in ``about.css``.
    """
    return [
        {"label": "Python 3.10+", "kind": ""},
        {"label": "FastAPI", "kind": "ac"},
        {"label": "Jinja2", "kind": "ac"},
        {"label": "HTMX 1.9", "kind": "ac"},
        {"label": "PyTorch", "kind": "yellow"},
        {"label": "NumPy", "kind": ""},
        {"label": "FFmpeg", "kind": "pink"},
        {"label": "PySceneDetect", "kind": "green"},
        {"label": "Babel · PT-BR / EN", "kind": ""},
    ]


def credits_list() -> list[dict[str, Any]]:
    """Return the institutional credits grid (label / value pairs).

    ``dim=True`` softens the value to ``--c-text-2`` (used for tertiary
    credits like model authors and acknowledgements).
    """
    return [
        {"role": "Concept", "name": "KUAA · Curatorial team", "dim": False},
        {"role": "Engineering", "name": "Rafael Perez", "dim": False},
        {
            "role": "AI integration",
            "name": "moondream, openai, ultralytics (model authors)",
            "dim": True,
        },
        {"role": "Funding", "name": "—", "dim": True},
        {"role": "Year", "name": "2026", "dim": False},
    ]
