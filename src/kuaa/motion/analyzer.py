"""Per-scene motion statistics from dense optical flow.

Why this exists
---------------
Every model in the system consumes a single still JPEG — SigLIP, YOLOv8,
MTCNN, Moondream and the environment heuristic all take one ``image_path``.
Downstream of ``scene_cuts.json`` a hard cut, a slow dissolve, a static
tableau and a whip pan are indistinguishable. The system's model of a film
is a bag of independent stills.

That is expensive on this corpus. On ``jeca_tatu_1959`` the median shot is
7.3s, the 90th percentile 25.2s, and the longest 186.4s — 36% of scenes run
over ten seconds and 114 run over thirty. A 186-second take is currently
represented by three still frames and nothing that says it barely moves.

What it measures
----------------
Dense Farnebäck flow between sampled frame pairs, decomposed into:

* **global** translation — the dominant flow shared by the whole frame,
  which is the camera moving;
* **residual** — flow left after removing the global component, which is
  subjects moving within the frame.

Keeping them apart is what makes the signal crossable with object
detection: ``horse`` + ``subject_motion: moving`` is a galloping horse,
``horse`` + ``static`` is a tableau. That crossing is where archive
vocabulary actually lives, and neither detector nor describer can express
it from a still.

What it does *not* do
---------------------
It does not feed the describer. ``llm_description`` is a sibling of
``visual_analysis`` and ``embeddings`` — all three depend only on
``scene_detection`` and none consumes another's output. Making the describer
depend on motion would invert that dependency, force a full LLM re-run, and
hand a still-image VLM a derived claim it cannot verify. Motion enters
retrieval directly instead, as tags and filters.

Cost: one decode pass per film, CPU-only, at ``sample_fps`` rather than
every frame.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Frames per second actually decoded. Camera moves and subject action are
# low-frequency relative to frame rate; sampling ~4fps captures both at a
# fraction of the decode cost.
DEFAULT_SAMPLE_FPS: float = 4.0

# Longest edge the analysis frame is scaled to. Farnebäck cost is quadratic
# in resolution and the statistics we want are global, so a small frame is
# both faster and less sensitive to grain — which matters on scanned film.
ANALYSIS_WIDTH: int = 320

# Flow magnitude (px/frame at ANALYSIS_WIDTH) below which a scene is called
# static. Archival scans carry grain and gate weave that register as a
# fraction of a pixel even on a locked-off shot.
STATIC_MAGNITUDE: float = 0.35

# Share of total flow explained by the global component, above which the
# motion is attributed to the camera rather than to subjects.
CAMERA_DOMINANCE: float = 0.60

# Minimum global magnitude before a camera move is named at all.
CAMERA_MIN_MAGNITUDE: float = 0.30


@dataclass(frozen=True)
class SceneMotion:
    """Motion statistics for one scene.

    Attributes:
        scene_id: the scene these statistics describe.
        shot_length_s: duration, carried here so the whole motion story
            for a scene is readable from one record.
        motion_magnitude: mean total flow magnitude, normalised to
            ``[0, 1]`` against a saturating reference.
        camera_motion: ``static`` | ``pan`` | ``tilt`` | ``zoom`` | ``complex``.
        subject_motion: ``still`` | ``moving``.
        camera_share: fraction of flow explained by global translation.
        sampled_pairs: how many frame pairs were measured. ``0`` means the
            scene could not be decoded and every other field is a default.
    """

    scene_id: int
    shot_length_s: float
    motion_magnitude: float
    camera_motion: str
    subject_motion: str
    camera_share: float
    sampled_pairs: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, rounded for a readable artefact."""
        d = asdict(self)
        for key in ("shot_length_s", "motion_magnitude", "camera_share"):
            d[key] = round(float(d[key]), 4)
        return d

    def tags(self) -> list[str]:
        """Portuguese kebab-case tags for the BM25 corpus and facet filters.

        Portuguese because the interface language is pt-BR and these are
        curator-facing; the bilingual expansion carries them to English.
        Only meaningful states produce a tag — a scene with no measurable
        motion signal contributes nothing rather than a misleading label.
        """
        if self.sampled_pairs == 0:
            return []
        out: list[str] = []
        if self.camera_motion == "static":
            out.append("plano-estatico")
        elif self.camera_motion == "pan":
            out.append("camera-panoramica")
        elif self.camera_motion == "tilt":
            out.append("camera-vertical")
        elif self.camera_motion == "zoom":
            out.append("camera-zoom")
        elif self.camera_motion == "complex":
            out.append("camera-movel")
        out.append("movimento-de-cena" if self.subject_motion == "moving" else "cena-parada")
        if self.motion_magnitude >= 0.6:
            out.append("movimento-intenso")
        if self.shot_length_s >= 30.0:
            out.append("plano-longo")
        return out


def _classify_camera(
    global_dx: float, global_dy: float, divergence: float, magnitude: float, camera_share: float
) -> str:
    """Name the dominant camera move from the decomposed flow."""
    if magnitude < STATIC_MAGNITUDE:
        return "static"
    if camera_share < CAMERA_DOMINANCE:
        # Flow is not dominated by a single global translation — subjects
        # moving independently, or a move too compound to name.
        return "complex"
    gx, gy = abs(global_dx), abs(global_dy)
    if max(gx, gy) < CAMERA_MIN_MAGNITUDE:
        return "zoom" if divergence >= CAMERA_MIN_MAGNITUDE else "static"
    # A zoom shows radial divergence rather than a consistent translation.
    if divergence > max(gx, gy):
        return "zoom"
    return "pan" if gx >= gy else "tilt"


def _scene_motion_from_flows(
    scene_id: int,
    shot_length_s: float,
    flows: list[np.ndarray],
) -> SceneMotion:
    """Reduce a scene's flow fields to one statistics record."""
    if not flows:
        return SceneMotion(
            scene_id=scene_id,
            shot_length_s=shot_length_s,
            motion_magnitude=0.0,
            camera_motion="static",
            subject_motion="still",
            camera_share=0.0,
            sampled_pairs=0,
        )

    total_mags: list[float] = []
    residual_mags: list[float] = []
    gxs: list[float] = []
    gys: list[float] = []
    divs: list[float] = []
    for flow in flows:
        fx, fy = flow[..., 0], flow[..., 1]
        # The global component is the median translation — median rather
        # than mean so a large moving subject cannot drag the estimate.
        gx = float(np.median(fx))
        gy = float(np.median(fy))
        total_mags.append(float(np.mean(np.hypot(fx, fy))))
        residual_mags.append(float(np.mean(np.hypot(fx - gx, fy - gy))))
        gxs.append(gx)
        gys.append(gy)
        # Radial divergence: does flow point outward from the centre (zoom
        # in) or inward (zoom out)? Measured as the correlation between
        # flow and position relative to centre.
        h, w = fx.shape
        ys, xs = np.mgrid[0:h, 0:w]
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        rx, ry = xs - cx, ys - cy
        norm = float(np.sqrt(np.mean(rx**2 + ry**2))) or 1.0
        divs.append(float(np.mean((fx - gx) * rx + (fy - gy) * ry) / norm))

    total = float(np.mean(total_mags))
    residual = float(np.mean(residual_mags))
    global_dx = float(np.mean(gxs))
    global_dy = float(np.mean(gys))
    divergence = abs(float(np.mean(divs)))
    camera_share = 0.0 if total <= 1e-6 else max(0.0, 1.0 - (residual / total))

    camera = _classify_camera(global_dx, global_dy, divergence, total, camera_share)
    subject = "moving" if residual >= STATIC_MAGNITUDE else "still"
    # Saturating normalisation: 6 px/frame at 320px wide is already a fast
    # move, so anything beyond it reads as 1.0 rather than compressing the
    # useful low end.
    magnitude = float(min(1.0, total / 6.0))

    return SceneMotion(
        scene_id=scene_id,
        shot_length_s=shot_length_s,
        motion_magnitude=magnitude,
        camera_motion=camera,
        subject_motion=subject,
        camera_share=camera_share,
        sampled_pairs=len(flows),
    )


def analyze_video(
    video_path: Path,
    boundaries: list[tuple[int, int]],
    fps: float,
    *,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    max_pairs_per_scene: int = 24,
) -> list[SceneMotion]:
    """Compute per-scene motion statistics for one video.

    Args:
        video_path: the source video.
        boundaries: ``[(start_frame, end_frame), ...]``, as produced by
            :meth:`kuaa.scene_detector.CutSet.scene_boundaries`. Scene ids
            are 1-based positions in this list, matching the rest of the
            pipeline.
        fps: frames per second of the source.
        sample_fps: how many frames per second to actually decode.
        max_pairs_per_scene: cost cap for very long takes — a 186-second
            tableau does not need 700 flow fields to be called static.

    Returns:
        One :class:`SceneMotion` per boundary, in scene order. A scene that
        could not be decoded comes back with ``sampled_pairs=0`` rather
        than raising, so one bad region cannot fail a whole film.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"could not open video: {video_path}")

    step = max(1, int(round(fps / max(sample_fps, 0.1))))
    out: list[SceneMotion] = []
    try:
        for scene_id, (start, end) in enumerate(boundaries, start=1):
            shot_length_s = max(0.0, (end - start) / fps) if fps else 0.0
            frame_positions = list(range(start, max(start + 1, end), step))
            if len(frame_positions) > max_pairs_per_scene + 1:
                # Even coverage of the scene rather than only its opening.
                idx = np.linspace(0, len(frame_positions) - 1, max_pairs_per_scene + 1)
                frame_positions = [frame_positions[int(i)] for i in idx]

            flows: list[np.ndarray] = []
            prev_gray = None
            for pos in frame_positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ok, frame = cap.read()
                if not ok:
                    break
                gray = _to_analysis_gray(cv2, frame)
                if prev_gray is not None:
                    # `flow=None` asks OpenCV to allocate the output itself —
                    # the documented calling form. opencv-python's stub types
                    # that parameter as a required array, so mypy rejects a
                    # call the runtime accepts.
                    flows.append(
                        cv2.calcOpticalFlowFarneback(  # type: ignore[call-overload]
                            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                        )
                    )
                prev_gray = gray
            out.append(_scene_motion_from_flows(scene_id, shot_length_s, flows))
    finally:
        cap.release()
    return out


def _to_analysis_gray(cv2_mod: Any, frame: np.ndarray) -> np.ndarray:
    """Downscale to :data:`ANALYSIS_WIDTH` and convert to greyscale."""
    h, w = frame.shape[:2]
    if w > ANALYSIS_WIDTH:
        scale = ANALYSIS_WIDTH / float(w)
        frame = cv2_mod.resize(frame, (ANALYSIS_WIDTH, max(1, int(round(h * scale)))))
    return cv2_mod.cvtColor(frame, cv2_mod.COLOR_BGR2GRAY)


def motion_tag_index(records: list[SceneMotion]) -> dict[str, list[int]]:
    """Invert motion records into the ``{tag: [scene_id, ...]}`` shape.

    Matches ``scene_tags.json`` so the motion artefact can join the BM25
    corpus and the facet filters through the existing tag machinery rather
    than a parallel path.
    """
    index: dict[str, list[int]] = {}
    for record in records:
        for tag in record.tags():
            index.setdefault(tag, []).append(record.scene_id)
    return index
