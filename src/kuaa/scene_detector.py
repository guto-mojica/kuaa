"""
kuaa.scene_detector
~~~~~~~~~~~~~~~~~~~~~~~~~
Detecção de cortes de cena e extração de keyframes usando PySceneDetect.

Baseado no Notebook 02 (02_deteccao_cenas.ipynb).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kuaa.config import _Namespace

logger = logging.getLogger(__name__)

# PySceneDetect — importação diferida para não quebrar se não instalado
try:
    from scenedetect import FrameTimecode, SceneManager, open_video
    from scenedetect.detectors import AdaptiveDetector, ContentDetector
    from scenedetect.scene_manager import save_images

    _SCENEDETECT_AVAILABLE = True
except ImportError:
    _SCENEDETECT_AVAILABLE = False
    logger.warning("PySceneDetect não instalado. Instale com: pip install scenedetect[opencv]")


SceneList = list[tuple]  # list of (FrameTimecode, FrameTimecode)


def _boundaries_to_scene_list(boundaries: list[tuple[int, int]], fps: float) -> SceneList:
    """Turn ``[(start_frame, end_frame), ...]`` into ``FrameTimecode`` pairs.

    Shared by :meth:`SceneDetector.cuts_to_scene_list` (a whole cut set) and
    :meth:`SceneDetector.extract_keyframes_for_boundaries` (an arbitrary
    boundary subset), so both flow through the same PySceneDetect-facing
    shape.
    """
    if not _SCENEDETECT_AVAILABLE:
        raise RuntimeError("PySceneDetect não instalado.")
    return [(FrameTimecode(start, fps), FrameTimecode(end, fps)) for start, end in boundaries]


# ─── Cut-list model ──────────────────────────────────────────────────────────
#
# The authoritative representation of scene boundaries is a list of *interior*
# cut frames. Scenes are derived as [(0, c1), (c1, c2), ..., (cn, total)], so
# both re-tuning (auto re-detect) and hand-correction (manual split / merge)
# become pure functions of ``(video, cut list)``. This is what the
# Pre-processing review UI reads and edits; it is persisted to
# ``metadata/scene_cuts.json`` alongside the keyframe artifacts.


@dataclass
class SceneCut:
    """One interior scene boundary.

    ``source`` is ``"auto"`` (produced by the detector) or ``"manual"``
    (added/kept by an operator during review).
    """

    frame: int
    time_s: float
    source: str = "auto"


@dataclass
class CutSet:
    """The effective scene-boundary set for a single film.

    ``cuts`` holds only the *interior* boundaries; the implicit 0 and
    ``total_frames`` bookends are added by :meth:`scene_boundaries`.
    """

    fps: float
    total_frames: int
    duration_s: float
    cuts: list[SceneCut] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def has_manual_edits(self) -> bool:
        return any(c.source == "manual" for c in self.cuts)

    @property
    def num_scenes(self) -> int:
        return len(self.cuts) + 1

    def sorted_cuts(self) -> list[SceneCut]:
        return sorted(self.cuts, key=lambda c: c.frame)

    def scene_boundaries(self) -> list[tuple[int, int]]:
        """Return ``[(start_frame, end_frame), ...]`` for every scene."""
        pts = [0, *[c.frame for c in self.sorted_cuts()], self.total_frames]
        return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration_s": self.duration_s,
            "has_manual_edits": self.has_manual_edits,
            "params": self.params,
            "cuts": [
                {"frame": c.frame, "time_s": c.time_s, "source": c.source}
                for c in self.sorted_cuts()
            ],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CutSet:
        cuts = [
            SceneCut(
                frame=int(c["frame"]),
                time_s=float(c.get("time_s", 0.0)),
                source=str(c.get("source", "auto")),
            )
            for c in raw.get("cuts", [])
        ]
        return cls(
            fps=float(raw.get("fps", 24.0)),
            total_frames=int(raw.get("total_frames", 0)),
            duration_s=float(raw.get("duration_s", 0.0)),
            cuts=cuts,
            params=dict(raw.get("params", {})),
        )


class SceneDetector:
    """
    Detecta cenas em um arquivo de vídeo e extrai keyframes representativos.

    Suporta dois algoritmos:
        "content"  — ContentDetector: diferença de histograma entre frames.
                     Bom para cortes diretos (hard cuts).
        "adaptive" — AdaptiveDetector: threshold adaptativo baseado em
                     variações locais. Melhor para dissolvências (fades).

    Exemplo:
        detector = SceneDetector(cfg)
        scenes = detector.detect("data/raw/jeca_tatu.mp4")
        keyframes = detector.extract_keyframes(scenes, "data/raw/jeca_tatu.mp4",
                                               "data/frames/scenes")
    """

    def __init__(self, cfg: _Namespace | None = None):
        if cfg is not None:
            sd = cfg.scene_detection
            self.detector_type = sd.detector
            self.content_threshold = sd.content_threshold
            self.adaptive_threshold = sd.adaptive_threshold
            self.min_scene_len = sd.min_scene_len
            self.keyframes_per_scene = sd.keyframes_per_scene
            self.keyframe_height = getattr(sd, "keyframe_height", 480)
        else:
            self.detector_type = "adaptive"
            self.content_threshold = 27.0
            self.adaptive_threshold = 3.0
            self.min_scene_len = 15
            self.keyframes_per_scene = 3
            self.keyframe_height = 480

    def detect(self, video_path: str | Path) -> SceneList:
        """
        Detecta cenas no vídeo.

        Args:
            video_path: Caminho do arquivo de vídeo.

        Returns:
            Lista de tuplas (start_timecode, end_timecode).

        Raises:
            RuntimeError: Se PySceneDetect não estiver instalado.
            FileNotFoundError: Se o vídeo não existir.
        """
        if not _SCENEDETECT_AVAILABLE:
            raise RuntimeError(
                "PySceneDetect não instalado. Execute: pip install scenedetect[opencv]"
            )

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

        video_manager = open_video(str(video_path))
        scene_manager = SceneManager()

        if self.detector_type == "adaptive":
            scene_manager.add_detector(
                AdaptiveDetector(
                    adaptive_threshold=self.adaptive_threshold,
                    min_scene_len=self.min_scene_len,
                )
            )
            logger.info(
                "Detectando cenas com AdaptiveDetector (threshold=%.1f, min_len=%d)",
                self.adaptive_threshold,
                self.min_scene_len,
            )
        else:
            scene_manager.add_detector(
                ContentDetector(
                    threshold=self.content_threshold,
                    min_scene_len=self.min_scene_len,
                )
            )
            logger.info(
                "Detectando cenas com ContentDetector (threshold=%.1f, min_len=%d)",
                self.content_threshold,
                self.min_scene_len,
            )

        scene_manager.detect_scenes(video_manager, show_progress=True)
        scene_list = scene_manager.get_scene_list()
        video_manager.capture.release()

        logger.info("✓ %d cenas detectadas em %s", len(scene_list), video_path.name)
        return scene_list

    def extract_keyframes(
        self,
        scene_list: SceneList,
        video_path: str | Path,
        output_dir: str | Path,
    ) -> list[Path]:
        """
        Extrai keyframes representativos de cada cena.

        Args:
            scene_list: Saída de self.detect().
            video_path: Caminho do vídeo original.
            output_dir: Diretório onde os keyframes serão salvos.

        Returns:
            Lista ordenada dos Paths dos keyframes extraídos.
        """
        if not _SCENEDETECT_AVAILABLE:
            raise RuntimeError("PySceneDetect não instalado.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_manager = open_video(str(video_path))
        save_images(
            scene_list,
            video_manager,
            num_images=self.keyframes_per_scene,
            image_extension="jpg",
            output_dir=str(output_dir),
            height=self.keyframe_height if self.keyframe_height > 0 else None,
        )
        video_manager.capture.release()

        keyframes = sorted(output_dir.glob("*.jpg"))
        logger.info("✓ %d keyframes extraídos em %s", len(keyframes), output_dir)
        return keyframes

    def scene_stats(self, scene_list: SceneList) -> dict:
        """
        Calcula estatísticas descritivas sobre as durações das cenas.

        Args:
            scene_list: Saída de self.detect().

        Returns:
            dict com num_scenes, total/mean/median/min/max/std de duração em segundos.
        """
        import numpy as np

        if not scene_list:
            return {}

        durations = np.array([(end - start).get_seconds() for start, end in scene_list])

        return {
            "num_scenes": len(scene_list),
            "total_duration_s": float(durations.sum()),
            "mean_s": float(durations.mean()),
            "median_s": float(np.median(durations)),
            "min_s": float(durations.min()),
            "max_s": float(durations.max()),
            "std_s": float(durations.std()),
        }

    def export_metadata(
        self,
        scene_list: SceneList,
        keyframe_paths: list[Path],
        output_path: str | Path,
    ) -> Path:
        """
        Exporta metadados das cenas para JSON.

        O JSON gerado inclui scene_id, timecodes e path do keyframe,
        compatível com o formato esperado pelos módulos seguintes.

        Args:
            scene_list:     Saída de self.detect().
            keyframe_paths: Saída de self.extract_keyframes().
            output_path:    Caminho do arquivo JSON de saída.

        Returns:
            Path do arquivo JSON criado.
        """
        # Emit one metadata row per saved keyframe (1:N), not per scene.
        # PySceneDetect already wrote N keyframes per scene to disk
        # (positions distributed across the scene's duration); the prior
        # 1:1 logic picked only the middle one and discarded the rest,
        # which under-represents long scenes — a 5.5-minute tableau shot
        # ended up as a single CLIP vector. Embedding all N triples the
        # index density at zero extraction cost.
        #
        # The N rows for scene `idx` share scene_id and the scene-level
        # time/frame fields; they differ in keyframe_id ("scene_NNNN_kf_KK")
        # and filepath. Downstream code (search) deduplicates back to one
        # result per scene at query time using max(similarity).
        kf_per_scene = max(1, self.keyframes_per_scene)
        scenes_data = []

        for idx, (start, end) in enumerate(scene_list):
            scene_id = idx + 1
            scene_block = keyframe_paths[idx * kf_per_scene : (idx + 1) * kf_per_scene]
            for kf_pos, kf_path in enumerate(scene_block, start=1):
                scenes_data.append(
                    {
                        "scene_id": scene_id,
                        "keyframe_id": f"scene_{scene_id:04d}_kf_{kf_pos:02d}",
                        "filepath": str(kf_path),
                        "start_time_s": start.get_seconds(),
                        "end_time_s": end.get_seconds(),
                        "duration_s": (end - start).get_seconds(),
                        "start_frame": start.get_frames(),
                        "end_frame": end.get_frames(),
                    }
                )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(scenes_data, f, indent=2, ensure_ascii=False)

        logger.info(
            "✓ Metadados de %d cenas exportados: %s",
            len(scenes_data),
            output_path,
        )
        return output_path

    # ─── Cut-list API ─────────────────────────────────────────────────────────

    def current_params(self) -> dict[str, Any]:
        """Snapshot the detector settings that produced (or will produce) a cut set."""
        return {
            "detector": self.detector_type,
            "content_threshold": self.content_threshold,
            "adaptive_threshold": self.adaptive_threshold,
            "min_scene_len": self.min_scene_len,
            "keyframes_per_scene": self.keyframes_per_scene,
            "keyframe_height": self.keyframe_height,
        }

    def build_cutset(self, scene_list: SceneList) -> CutSet:
        """Derive a :class:`CutSet` from a PySceneDetect scene list.

        The interior cut frames are the end of every scene except the last;
        ``total_frames`` is the end of the last scene. FPS is read from the
        timecode framerate, falling back to frames/seconds.
        """
        if not scene_list:
            return CutSet(fps=24.0, total_frames=0, duration_s=0.0, params=self.current_params())

        last_end = scene_list[-1][1]
        total_frames = int(last_end.get_frames())
        duration_s = float(last_end.get_seconds())
        fps = _framerate_of(last_end, duration_s, total_frames)

        cuts = [
            SceneCut(
                frame=int(end.get_frames()),
                time_s=float(end.get_seconds()),
                source="auto",
            )
            for start, end in scene_list[:-1]
        ]
        return CutSet(
            fps=fps,
            total_frames=total_frames,
            duration_s=duration_s,
            cuts=cuts,
            params=self.current_params(),
        )

    def detect_cuts(self, video_path: str | Path) -> CutSet:
        """Run detection and return the authoritative :class:`CutSet`."""
        scene_list = self.detect(video_path)
        return self.build_cutset(scene_list)

    def cuts_to_scene_list(self, cutset: CutSet) -> SceneList:
        """Reconstruct a PySceneDetect scene list from a cut set.

        Builds ``FrameTimecode`` bookends so the reconstructed scenes flow
        back through the proven :meth:`extract_keyframes` /
        :meth:`export_metadata` path unchanged.
        """
        return _boundaries_to_scene_list(cutset.scene_boundaries(), cutset.fps or 24.0)

    def rebuild(
        self,
        cutset: CutSet,
        video_path: str | Path,
        keyframes_dir: str | Path,
        metadata_path: str | Path,
    ) -> tuple[list[Path], Path]:
        """Regenerate keyframes + ``keyframes_metadata.json`` from a cut set.

        Clears ``keyframes_dir`` first so a shorter cut list never leaves
        orphaned images behind, then re-extracts every keyframe and rewrites
        the metadata. Used by auto re-detect (a fresh run has no "unchanged"
        scenes to preserve) — manual split/merge instead go through the
        Pre-processing review's staged-edit path
        (:func:`kuaa.preprocess.service.apply_pending`), which only
        re-extracts the scenes an edit actually touched and renames the rest
        via :meth:`extract_keyframes_for_boundaries`.
        """
        keyframes_dir = Path(keyframes_dir)
        if keyframes_dir.exists():
            for old in keyframes_dir.glob("*.jpg"):
                old.unlink()

        scene_list = self.cuts_to_scene_list(cutset)
        keyframes = self.extract_keyframes(scene_list, video_path, keyframes_dir)
        out = self.export_metadata(scene_list, keyframes, metadata_path)
        return keyframes, out

    def extract_keyframes_for_boundaries(
        self,
        boundaries: list[tuple[int, int]],
        scene_ids: list[int],
        fps: float,
        video_path: str | Path,
        output_dir: str | Path,
    ) -> dict[int, list[Path]]:
        """Extract + canonically name keyframes for a boundary subset only.

        Used by the Pre-processing "Apply changes" partial rebuild: only the
        scene(s) actually touched by a staged split/merge need a fresh video
        decode. ``scene_ids[i]`` is the FINAL absolute scene number for
        ``boundaries[i]`` (its position in the whole film after the edit) —
        used to name the output files, since ``save_images`` always numbers
        its own output starting at 1 for whatever subset it is given,
        unrelated to a scene's true position in the film.

        Returns ``{scene_id: [ordered keyframe Paths]}``.
        """
        if not _SCENEDETECT_AVAILABLE:
            raise RuntimeError("PySceneDetect não instalado.")
        if len(boundaries) != len(scene_ids):
            raise ValueError("boundaries e scene_ids devem ter o mesmo tamanho.")
        if not boundaries:
            return {}

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        scene_list = _boundaries_to_scene_list(boundaries, fps or 24.0)
        video_manager = open_video(str(video_path))
        raw = save_images(
            scene_list,
            video_manager,
            num_images=self.keyframes_per_scene,
            image_extension="jpg",
            output_dir=str(output_dir),
            height=self.keyframe_height if self.keyframe_height > 0 else None,
        )
        video_manager.capture.release()

        # `raw` keys are 0-based positions within `scene_list`, matching
        # `boundaries`/`scene_ids` order 1:1 (confirmed against PySceneDetect's
        # source — its docstring says "starting from 1" but the actual
        # implementation enumerates from 0). Values are bare filenames
        # relative to `output_dir`, not full paths.
        out: dict[int, list[Path]] = {}
        for local_idx, scene_id in enumerate(scene_ids):
            filenames = raw.get(local_idx, [])
            renamed: list[Path] = []
            for pos, filename in enumerate(filenames, start=1):
                src = output_dir / filename
                dest = output_dir / f"scene_{scene_id:04d}_kf_{pos:02d}.jpg"
                src.rename(dest)
                renamed.append(dest)
            out[scene_id] = renamed
        logger.info("✓ %d cenas extraídas (parcial) em %s", len(scene_ids), output_dir)
        return out


def write_cutset(cutset: CutSet, path: str | Path) -> Path:
    """Persist a cut set to ``scene_cuts.json``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cutset.to_dict(), f, indent=2, ensure_ascii=False)
    return path


def read_cutset(path: str | Path) -> CutSet | None:
    """Load a cut set from ``scene_cuts.json``, or ``None`` if absent."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return CutSet.from_dict(json.load(f))


def _framerate_of(timecode: Any, duration_s: float, total_frames: int) -> float:
    """Best-effort FPS from a FrameTimecode, falling back to frames/seconds."""
    getter = getattr(timecode, "get_framerate", None)
    if callable(getter):
        try:
            fps = float(getter())
            if fps > 0:
                return fps
        except Exception:  # pragma: no cover - defensive
            pass
    if duration_s > 0 and total_frames > 0:
        return total_frames / duration_s
    return 24.0
