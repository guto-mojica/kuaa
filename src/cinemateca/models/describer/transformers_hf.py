"""
cinemateca.models.describer.transformers_hf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Default scene describer: Moondream 2 via Hugging Face transformers.

Loads ``vikhyatk/moondream2`` at the configured revision (default
``2025-01-09``) with ``trust_remote_code``. GPU acceleration comes from
the installed PyTorch wheel (CUDA / Apple MPS) — no source build. The
2025-01-09 remote code is transformers-4 only; the dependency is pinned
``transformers>=4.44,<5`` in pyproject (verified 2026-05-18: tf5 hard-fails
for every moondream2 revision).

Parsing, tagging and resume behaviour are shared verbatim with every other
describer backend via :mod:`cinemateca.models.describer._common`, so
switching backends does not change metadata semantics.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

import pandas as pd

from cinemateca.models.describer._common import (
    PROMPTS,
    build_metadata,
)

logger = logging.getLogger(__name__)

# Moondream 2 uses a SigLIP encoder with 378x378 input. Pre-resizing with
# PIL (fast, bilinear) avoids the model's internal high-quality multi-pass
# resize (~20s/frame) for the same effective input.
_INPUT_SIZE = 378


class MoondreamTransformersDescriber:
    """SceneDescriber backed by Moondream 2 via HF transformers."""

    def __init__(self, cfg=None, device=None):
        self._model = None
        self._tokenizer = None
        self._enc_cache: tuple[str, object] | None = None
        self._device = device
        if cfg is not None and getattr(cfg, "llm", None) is not None:
            self.model_id = cfg.llm.model_id
            self.revision = cfg.llm.revision
            self.checkpoint_interval = cfg.llm.checkpoint_interval
            self.process_limit = cfg.llm.process_limit
            self.descriptions_filename = cfg.llm.descriptions_filename
            self.tags_filename = cfg.llm.tags_filename
        else:
            self.model_id = "vikhyatk/moondream2"
            self.revision = "2025-01-09"
            self.checkpoint_interval = 25
            self.process_limit = None
            self.descriptions_filename = "scene_descriptions.json"
            self.tags_filename = "scene_tags.json"

    def _warn_if_cpu_torch(self) -> None:
        """Loud, self-announcing diagnostic for the silent CPU regression.

        An NVIDIA GPU is present but the installed PyTorch is a CPU-only
        wheel, so description runs ~10-25x slower with no other symptom.
        Mirrors gguf.py:_warn_if_cpu_build. Never raises: a diagnostic
        must not break model loading.
        """
        if not shutil.which("nvidia-smi"):
            return  # no NVIDIA GPU → CPU is expected, not a regression
        try:
            import torch

            cuda_ok = torch.cuda.is_available()
        except Exception:  # noqa: BLE001 - diagnostic must never break loading
            return
        if not cuda_ok:
            logger.warning(
                "Há GPU NVIDIA mas o PyTorch instalado é build CPU-only — a "
                "descrição roda ~10-25x mais devagar. Instale o torch CUDA: "
                "pip install torch --index-url "
                "https://download.pytorch.org/whl/cu128 (ajuste a versão CUDA)."
            )

    def _load_model(self):
        if self._model is not None:
            return
        self._warn_if_cpu_torch()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # noqa: TRY003
            raise RuntimeError("transformers não instalado. Rode: uv sync --extra full") from e
        import torch

        # float16 on GPU/MPS (2x faster, half the memory); float32 on CPU
        # (CPUs lack native float16 SIMD). torch.device → use .type so
        # "cuda:0" is handled (str() compare would miss it).
        on_accel = self._device is not None and self._device.type in ("cuda", "mps")
        dtype = torch.float16 if on_accel else torch.float32

        logger.info(
            "Carregando Moondream 2 (%s, rev=%s) — dtype=%s device=%s — "
            "primeira execução baixa ~1.9GB...",
            self.model_id,
            self.revision,
            dtype,
            getattr(self._device, "type", "cpu"),
        )
        t0 = time.time()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        if self._device is not None:
            self._model = self._model.to(self._device)
        self._model.eval()
        logger.info("✓ Moondream 2 carregado em %.1fs", time.time() - t0)

    def _encode(self, image_path):
        """Open, RGB, pre-resize, run the SigLIP encoder once per frame.

        Single-entry cache keyed by resolved path: describe()/describe_batch
        ask 6 prompts about the same image; the encoder must run once.
        """
        key = str(Path(image_path).resolve())
        if self._enc_cache is not None and self._enc_cache[0] == key:
            return self._enc_cache[1]
        from PIL import Image

        img = (
            Image.open(image_path)
            .convert("RGB")
            .resize((_INPUT_SIZE, _INPUT_SIZE), Image.Resampling.BILINEAR)
        )
        enc = self._model.encode_image(img)
        self._enc_cache = (key, enc)
        return enc

    def _answer(self, image_path, prompt: str, max_tokens: int) -> str:
        """One image+prompt -> stripped model text (encoder cached per frame)."""
        enc = self._encode(image_path)
        return self._model.answer_question(
            enc, prompt, self._tokenizer, max_new_tokens=max_tokens
        ).strip()

    def describe(self, image_path) -> dict:
        """Return full metadata dict for a single keyframe."""
        self._load_model()
        raw = {}
        for field, (prompt, max_tokens) in PROMPTS.items():
            try:
                raw[field] = self._answer(image_path, prompt, max_tokens)
            except Exception as e:  # noqa: BLE001 - per-field resilience
                raw[field] = f"ERROR: {e}"
        row = pd.Series({"filepath": str(image_path)})
        return build_metadata(row, raw)

    def describe_batch(
        self,
        keyframes_df: pd.DataFrame,
        existing_results: list | None = None,
        checkpoint_path: Path | None = None,
    ) -> list[dict]:
        """Process all rows; resume via existing_results.

        RESUME-BUG FIX (mirrors gguf.py): error rows are NOT counted as
        processed — they are dropped so reprocessing can produce a good
        result; good rows are preserved verbatim, not rebuilt.
        """
        existing = list(existing_results or [])
        processed_ids = {r["scene_id"] for r in existing if "error" not in r}
        all_results = [r for r in existing if "error" not in r]
        to_process = keyframes_df[~keyframes_df["scene_id"].isin(processed_ids)].reset_index(
            drop=True
        )
        if self.process_limit:
            to_process = to_process.head(self.process_limit)

        logger.info(
            "LLM(transformers): %d a processar (%d já ok, %d total)",
            len(to_process),
            len(processed_ids),
            len(keyframes_df),
        )

        for count, (_, row) in enumerate(to_process.iterrows(), start=1):
            try:
                raw = {}
                self._load_model()
                for field, (prompt, mx) in PROMPTS.items():
                    try:
                        raw[field] = self._answer(row["filepath"], prompt, mx)
                    except Exception as e:  # noqa: BLE001
                        raw[field] = f"ERROR: {e}"
                meta = build_metadata(row, raw)
                all_results.append(meta)
                logger.info(
                    "cena %s [%d/%d]: %s | tags=%s",
                    meta.get("scene_id"),
                    count,
                    len(to_process),
                    str(meta.get("description", ""))[:70],
                    meta.get("tags", []),
                )
            except Exception as e:  # noqa: BLE001 - whole-frame failure
                all_results.append(
                    {
                        "scene_id": int(row.get("scene_id", -1)),
                        "keyframe_path": str(row["filepath"]),
                        "error": str(e),
                        "tags": [],
                        "objects": [],
                    }
                )
                logger.error("Erro cena %s: %s", row.get("scene_id"), e)

            if checkpoint_path and count % self.checkpoint_interval == 0:
                self._save_json(all_results, checkpoint_path)
                logger.info("Checkpoint: %d/%d", count, len(to_process))

        return all_results

    @staticmethod
    def build_tag_index(results: list[dict]) -> dict:
        """Build a tag → [scene_id] index sorted by frequency descending."""
        from collections import defaultdict

        idx: dict[str, list] = defaultdict(list)
        for rec in results:
            for tag in rec.get("tags", []):
                idx[tag].append(rec.get("scene_id"))
        return dict(sorted(idx.items(), key=lambda x: len(x[1]), reverse=True))

    def save(self, results, tag_index, output_dir):
        """Persist results and tag index as JSON files under output_dir."""
        import json

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        desc_path = out / self.descriptions_filename
        tags_path = out / self.tags_filename
        self._save_json(results, desc_path)
        with open(tags_path, "w", encoding="utf-8") as f:
            json.dump(tag_index, f, indent=2, ensure_ascii=False)
        return desc_path, tags_path

    @staticmethod
    def _save_json(data, path: Path):
        """Atomically write data as indented JSON to path."""
        import json

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
