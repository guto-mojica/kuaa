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

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class ClapHFEmbedder:
    """HF-transformers CLAP audio embedder."""

    def __init__(self, cfg=None, device=None) -> None:
        self._cfg = cfg
        self._device = device
        self._model = None
        self._processor = None

        ae = getattr(cfg, "audio_embeddings", None) if cfg else None
        self._model_id = getattr(ae, "model_id", "laion/larger_clap_general") if ae else "laion/larger_clap_general"
        self._batch_size = int(getattr(ae, "batch_size", 8)) if ae else 8
        self._chunk_seconds = float(getattr(ae, "chunk_seconds", 10.0)) if ae else 10.0
        self._sample_rate = int(getattr(ae, "sample_rate", 48000)) if ae else 48000

    # ── Lazy load ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import ClapModel, ClapProcessor
        except ImportError as exc:
            raise RuntimeError(
                "CLAP backend requires the 'full' extras. "
                "Install with: uv sync --extra full"
            ) from exc

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Carregando CLAP: %s (device=%s)", self._model_id, self._device)
        self._processor = ClapProcessor.from_pretrained(self._model_id)
        self._model = ClapModel.from_pretrained(self._model_id).to(self._device).eval()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _chunk_audio(self, wav: np.ndarray, sr: int) -> list[np.ndarray]:
        """Split a 1-D waveform into non-overlapping ``chunk_seconds`` slices.

        The trailing partial chunk is kept if it is at least 1 sample long
        (CLAP's processor pads to the model's required length).
        """
        chunk_samples = int(self._chunk_seconds * sr)
        if wav.shape[0] <= chunk_samples:
            return [wav]
        chunks = []
        for start in range(0, wav.shape[0], chunk_samples):
            chunk = wav[start:start + chunk_samples]
            if chunk.shape[0] > 0:
                chunks.append(chunk)
        return chunks

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

    @staticmethod
    def _l2_normalise(v: np.ndarray) -> np.ndarray:
        """Row-wise L2-normalise. Zero rows are left as zero (no NaN)."""
        if v.ndim == 1:
            n = float(np.linalg.norm(v))
            return v if n == 0.0 else (v / n).astype("float32")
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (v / norms).astype("float32")

    # ── Public API (AudioEmbedder Protocol) ──────────────────────────────

    def encode_audio_single(self, wav_path: str | Path) -> np.ndarray:
        self._load_model()
        wav, sr = sf.read(str(wav_path), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)  # downmix to mono
        if sr != self._sample_rate:
            raise ValueError(
                f"WAV sample rate {sr} != configured {self._sample_rate}. "
                f"Re-extract via SceneAudioExtractor (which writes at the "
                f"configured rate) before encoding."
            )
        chunks = self._chunk_audio(wav, sr)
        chunk_vecs = self._embed_chunks(chunks)         # (n_chunks, D)
        pooled = chunk_vecs.mean(axis=0)                 # (D,)
        return self._l2_normalise(pooled)

    def encode_audio(self, wav_paths: list[Path]) -> np.ndarray:
        if not wav_paths:
            # 512 matches every CLAP variant in current LAION lineup; if a
            # future config swaps to a non-512 model, hoist this to read
            # ``self._model.config.projection_dim`` (requires _load_model()).
            return np.zeros((0, 512), dtype="float32")
        self._load_model()
        vecs = [self.encode_audio_single(p) for p in wav_paths]
        return np.stack(vecs, axis=0).astype("float32")

    def encode_text(self, text: str) -> np.ndarray:
        import torch

        self._load_model()
        with torch.no_grad():
            inputs = self._processor(text=[text], return_tensors="pt", padding=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            feats = self._model.get_text_features(**inputs)
        v = feats.detach().cpu().numpy().astype("float32").squeeze(0)
        return self._l2_normalise(v)

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

        ``rows`` carries one dict per row index, with keys
        ``scene_id`` / ``wav_path`` / ``start_time_s`` / ``end_time_s`` /
        ``chunks_per_scene``.
        """
        import json

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        emb_path = out / embeddings_filename
        np.save(emb_path, embeddings)
        logger.info("✓ CLAP embeddings: %s | %.2f MB",
                    emb_path, emb_path.stat().st_size / 1e6)

        mapping = {
            "model": self._model_id,
            "dimension": int(embeddings.shape[1]) if embeddings.size else 512,
            "total_vectors": int(len(embeddings)),
            "normalized": True,
            "scene_ids": [r["scene_id"] for r in rows],
            "wav_paths": [r["wav_path"] for r in rows],
            "start_times_s": [r["start_time_s"] for r in rows],
            "end_times_s": [r["end_time_s"] for r in rows],
            "chunks_per_scene": [r["chunks_per_scene"] for r in rows],
        }
        map_path = out / mapping_filename
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        logger.info("✓ Mapeamento de áudio: %s", map_path)
        return emb_path, map_path
