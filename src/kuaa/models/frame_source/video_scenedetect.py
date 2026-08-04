"""
kuaa.models.frame_source.video_scenedetect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Default frame source: PySceneDetect scene detection + keyframe extraction.

This backend is a thin wrapper around :class:`kuaa.scene_detector.SceneDetector`
that reproduces, byte-for-byte, the sequence the pipeline's
``scene_detection`` step ran inline before Seam 1
(detect → extract_keyframes → export_metadata → write_cutset(build_cutset)
→ scene_stats). Keeping the call order identical guarantees existing
artefacts and snapshot tests stay green.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from kuaa.models.base import KeyframeManifest

if TYPE_CHECKING:
    from kuaa.config import Settings


class VideoSceneDetectFrameSource:
    """FrameSource backed by PySceneDetect (the video default)."""

    def __init__(self, cfg: Settings, device=None) -> None:
        # Device is unused (scene detection is CPU/FFmpeg), accepted for a
        # uniform factory signature with the model backends.
        self.cfg = cfg

    def produce(
        self,
        source: str | Path,
        *,
        keyframes_dir: Path,
        metadata_path: Path,
        cuts_path: Path | None = None,
    ) -> KeyframeManifest:
        """Detect scenes in ``source`` and extract representative keyframes.

        Identical to the historical inline ``_step_scene_detection`` body:
        the returned manifest carries the same ``metadata_path`` /
        ``keyframes`` / ``keyframes_dir`` / ``stats`` the step has always
        exposed. When ``cuts_path`` is given the authoritative cut set is
        persisted for the Pre-processing review UI.
        """
        from kuaa.scene_detector import SceneDetector, write_cutset

        detector = SceneDetector(self.cfg)
        scene_list = detector.detect(source)
        keyframes = detector.extract_keyframes(scene_list, source, keyframes_dir)
        metadata_path = detector.export_metadata(scene_list, keyframes, metadata_path)
        if cuts_path is not None:
            write_cutset(detector.build_cutset(scene_list), cuts_path)
        stats = detector.scene_stats(scene_list)
        return {
            "metadata_path": metadata_path,
            "keyframes": keyframes,
            "keyframes_dir": keyframes_dir,
            "stats": stats,
        }
