"""
cinemateca.models.audio.clap_hf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LAION-CLAP audio embedder via HuggingFace transformers.

Loads ``laion/larger_clap_general`` (default; configurable via
``cfg.audio_embeddings.model_id``). Joint text + audio space, so a
single backend serves text-as-query and audio-as-query.

Long-scene handling
-------------------
CLAP processes ~10 seconds at 48 kHz natively (480 000 samples). Scenes
longer than ``cfg.audio_embeddings.chunk_seconds`` are split into
non-overlapping chunks, each encoded separately, then **mean-pooled and
L2-renormalised** to one vector per scene. Mean-pool is the conventional
choice; if M2 evaluation shows weakness on long scenes (e.g. a 4-minute
shot with one music cue near the end) we can revisit max-pool or
attention-pool in a follow-up.

Model load is lazy: instantiating the backend is cheap (no GPU init);
the first call to ``encode_*`` triggers ``_load_model()``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from cinemateca.config import Settings
from cinemateca.models.manifest import ModelCard, get_card

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "laion/larger_clap_general"
_DEFAULT_BATCH_SIZE = 8
_DEFAULT_CHUNK_SECONDS = 10.0
_DEFAULT_SAMPLE_RATE = 48000
# Joint embedding dimension shared by all LAION-CLAP variants in the current
# lineup.  Used as a zero-row fallback when no WAVs are supplied — the caller
# never has to hard-code "512".  If a future model uses a different dim the
# constant is the single place to update.
_CLAP_DIM = 512


class ClapHFEmbedder:
    """HF-transformers CLAP audio embedder."""

    #: Provenance for this backend (manifest single source of truth, C10/F6).
    CARD: ModelCard = get_card("clap_hf")

    def __init__(self, cfg: Settings | None = None, device=None) -> None:
        self._cfg = cfg
        self._device = device
        # transformers stubs are incomplete (no get_audio_features/etc.).
        self._model: Any = None
        self._processor: Any = None

        ae = getattr(cfg, "audio_embeddings", None) if cfg else None
        self._model_id = str(getattr(ae, "model_id", _DEFAULT_MODEL_ID))
        self._batch_size = int(getattr(ae, "batch_size", _DEFAULT_BATCH_SIZE))
        self._chunk_seconds = float(getattr(ae, "chunk_seconds", _DEFAULT_CHUNK_SECONDS))
        self._sample_rate = int(getattr(ae, "sample_rate", _DEFAULT_SAMPLE_RATE))

    # ── Lazy load ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import ClapModel, ClapProcessor
        except ImportError as exc:
            raise RuntimeError(
                "CLAP backend requires the 'full' extras. Install with: uv sync --extra full"
            ) from exc

        if self._device is None:
            from cinemateca.device import get_device

            self._device = str(get_device("auto"))

        logger.info("Carregando CLAP: %s (device=%s)", self._model_id, self._device)
        # model_id comes from project config, not user input (offline-only system).
        self._processor = ClapProcessor.from_pretrained(self._model_id)  # nosec B615
        self._model = ClapModel.from_pretrained(
            self._model_id, device_map=self._device
        ).eval()  # nosec B615

    # ── Helpers ───────────────────────────────────────────────────────────

    def _chunk_audio(self, wav: np.ndarray, sr: int) -> list[np.ndarray]:
        """Split a 1-D waveform into non-overlapping ``chunk_seconds`` slices.

        The trailing partial chunk is kept (CLAP's processor pads to the
        model's required length).
        """
        chunk_samples = int(self._chunk_seconds * sr)
        if wav.shape[0] <= chunk_samples:
            return [wav]
        return [
            wav[s : s + chunk_samples]
            for s in range(0, wav.shape[0], chunk_samples)
            if wav[s : s + chunk_samples].shape[0] > 0
        ]

    def _embed_chunks(self, chunks: list[np.ndarray]) -> np.ndarray:
        """Encode chunks with the underlying CLAP model. Returns (n_chunks, D)."""
        import torch

        with torch.no_grad():
            inputs = self._processor(
                audios=chunks,
                sampling_rate=self._sample_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            feats = self._model.get_audio_features(**inputs)
        return feats.detach().cpu().numpy().astype("float32")

    def _load_wav(self, wav_path: str | Path) -> np.ndarray:
        wav, sr = sf.read(str(wav_path), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != self._sample_rate:
            raise ValueError(
                f"WAV sample rate {sr} != configured {self._sample_rate}. "
                f"Re-extract via SceneAudioExtractor before encoding."
            )
        return wav

    # ── Public API (AudioEmbedder Protocol) ──────────────────────────────

    def encode_audio_single(self, wav_path: str | Path) -> np.ndarray:
        """Return (D,) float32 L2-normalised vector for one WAV (audio-by-audio search)."""
        self._load_model()
        chunks = self._chunk_audio(self._load_wav(wav_path), self._sample_rate)
        pooled = self._embed_chunks(chunks).mean(axis=0)
        norm = float(np.linalg.norm(pooled)) or 1.0
        return (pooled / norm).astype("float32")

    def encode_audio(self, wav_paths: list[Path]) -> np.ndarray:
        """Encode multiple WAVs. Chunks are batched across scenes to amortise
        GPU launch overhead — 412 scenes × 1 chunk each goes from 412 forward
        passes to ~52 (at batch_size=8).
        """
        if not wav_paths:
            # Use _CLAP_DIM so there's a single place to update when a future
            # model variant changes the projection dimension.
            return np.zeros((0, _CLAP_DIM), dtype="float32")

        self._load_model()

        all_chunks: list[np.ndarray] = []
        chunks_per_scene: list[int] = []
        for p in wav_paths:
            scene_chunks = self._chunk_audio(self._load_wav(p), self._sample_rate)
            all_chunks.extend(scene_chunks)
            chunks_per_scene.append(len(scene_chunks))

        chunk_vecs_parts: list[np.ndarray] = []
        for i in range(0, len(all_chunks), self._batch_size):
            chunk_vecs_parts.append(self._embed_chunks(all_chunks[i : i + self._batch_size]))
        chunk_vecs = np.vstack(chunk_vecs_parts) if chunk_vecs_parts else np.empty((0, _CLAP_DIM))

        out = np.empty((len(wav_paths), chunk_vecs.shape[1]), dtype="float32")
        offset = 0
        for row_idx, n in enumerate(chunks_per_scene):
            pooled = chunk_vecs[offset : offset + n].mean(axis=0)
            norm = float(np.linalg.norm(pooled)) or 1.0
            out[row_idx] = pooled / norm
            offset += n
        return out

    def encode_text(self, text: str) -> np.ndarray:
        """Return (D,) float32 L2-normalised vector for a text query (shared CLAP space)."""
        import torch

        self._load_model()
        with torch.no_grad():
            inputs = self._processor(text=[text], return_tensors="pt", padding=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            feats = self._model.get_text_features(**inputs)
        v = feats.detach().cpu().numpy().astype("float32").squeeze(0)
        norm = float(np.linalg.norm(v)) or 1.0
        return (v / norm).astype("float32")

    # ── Save helper (parallel to OpenClipEmbedder.save) ──────────────────

    def save(
        self,
        embeddings: np.ndarray,
        rows: list[dict],
        output_dir: str | Path,
        embeddings_filename: str = "clap_embeddings.npy",
        mapping_filename: str = "audio_mapping.json",
    ) -> tuple[Path, Path]:
        """Persist embeddings + parallel-array mapping JSON.

        ``rows`` carries one dict per row index with keys
        ``scene_id`` / ``wav_path`` / ``start_time_s`` / ``end_time_s``.
        """
        import json

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        emb_path = out / embeddings_filename
        np.save(emb_path, embeddings)
        logger.info("✓ CLAP embeddings: %s | %.2f MB", emb_path, emb_path.stat().st_size / 1e6)

        mapping = {
            "model": self._model_id,
            "dimension": int(embeddings.shape[1]) if embeddings.size else _CLAP_DIM,
            "total_vectors": int(len(embeddings)),
            "normalized": True,
            "scene_ids": [r["scene_id"] for r in rows],
            "wav_paths": [r["wav_path"] for r in rows],
            "start_times_s": [r["start_time_s"] for r in rows],
            "end_times_s": [r["end_time_s"] for r in rows],
        }
        map_path = out / mapping_filename
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        logger.info("✓ Mapeamento de áudio: %s", map_path)
        return emb_path, map_path
