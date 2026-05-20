"""
cinemateca.models.describer.gguf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Keyless, offline Moondream 2 scene describer via llama-cpp-python.
Loads the 2025-01-09 GGUF pair (text model + mmproj) from the
vikhyatk/moondream2 repo at that revision. No transformers, no API key.
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
from cinemateca.models.describer.domain_prompts import prompts_from_config

logger = logging.getLogger(__name__)

_REPO = "vikhyatk/moondream2"
_REV = "2025-01-09"
_TEXT_GGUF = "moondream2-text-model-f16.gguf"
_MMPROJ_GGUF = "moondream2-mmproj-f16.gguf"


class MoondreamGGUFDescriber:
    """SceneDescriber backed by Moondream 2 GGUF + llama-cpp-python."""

    def __init__(self, cfg=None, device=None):
        self._llm = None
        if cfg is not None and getattr(cfg, "llm", None) is not None:
            self.checkpoint_interval = cfg.llm.checkpoint_interval
            self.process_limit = cfg.llm.process_limit
            self.descriptions_filename = cfg.llm.descriptions_filename
            self.tags_filename = cfg.llm.tags_filename
            # -1 = offload every layer to GPU. Harmless on a CPU-only
            # llama-cpp-python build (no CUDA → it silently runs on CPU).
            self.n_gpu_layers = getattr(cfg.llm, "gpu_layers", -1)
            self.prompts = prompts_from_config(cfg)
        else:
            self.checkpoint_interval = 25
            self.process_limit = None
            self.descriptions_filename = "scene_descriptions.json"
            self.tags_filename = "scene_tags.json"
            self.n_gpu_layers = -1
            self.prompts = dict(PROMPTS)

    def _warn_if_cpu_build(self) -> None:
        """Loud, self-announcing diagnostic for the silent GPU regression.

        GPU offload was requested (``gpu_layers != 0``) and an NVIDIA GPU is
        present, but the installed ``llama-cpp-python`` is a CPU-only build —
        so runs are ~25x slower with no other symptom. This commonly happens
        because ``uv sync``/``uv run`` replaces the from-source CUDA build
        with the cached CPU wheel. Never raises: a diagnostic must not break
        model loading.
        """
        if self.n_gpu_layers == 0:
            return
        if not shutil.which("nvidia-smi"):
            return  # no NVIDIA GPU → a CPU-only build is expected, not a bug
        try:
            from llama_cpp import llama_cpp as _core

            gpu_ok = _core.llama_supports_gpu_offload()
        except Exception:  # noqa: BLE001 - diagnostic must never break loading
            return
        if not gpu_ok:
            logger.warning(
                "llm.gpu_layers=%d e há GPU NVIDIA, mas llama-cpp-python é um "
                "build CPU-only — a descrição roda ~25x mais devagar. "
                "Reconstrua com CUDA: docs/GPU_LLAMA_CPP_CUDA_BUILD.md "
                "(ou rode scripts/ensure_gpu_llama.sh).",
                self.n_gpu_layers,
            )

    def _load_model(self):
        if self._llm is not None:
            return
        self._warn_if_cpu_build()
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import MoondreamChatHandler

        t0 = time.time()
        text_path = hf_hub_download(_REPO, _TEXT_GGUF, revision=_REV)
        mmproj_path = hf_hub_download(_REPO, _MMPROJ_GGUF, revision=_REV)
        handler = MoondreamChatHandler(clip_model_path=mmproj_path)
        self._llm = Llama(
            model_path=text_path,
            chat_handler=handler,
            n_ctx=2048,
            logits_all=True,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )
        logger.info(
            "✓ Moondream GGUF carregado em %.1fs (n_gpu_layers=%d)",
            time.time() - t0,
            self.n_gpu_layers,
        )

    def _answer(self, image_path, prompt: str, max_tokens: int) -> str:
        """One image+prompt -> stripped model text. Re-embeds per call."""
        uri = Path(image_path).resolve().as_uri()
        resp = self._llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": uri}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=max_tokens,
        )
        return resp["choices"][0]["message"]["content"].strip()

    def describe(self, image_path) -> dict:
        self._load_model()
        raw = {}
        for field, (prompt, max_tokens) in self.prompts.items():
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
        # RESUME-BUG FIX: error rows are NOT counted as processed — they are
        # dropped so reprocessing can produce a good result (mirror pipeline.py:332).
        existing = list(existing_results or [])
        processed_ids = {r["scene_id"] for r in existing if "error" not in r}
        all_results = [r for r in existing if "error" not in r]
        to_process = keyframes_df[~keyframes_df["scene_id"].isin(processed_ids)].reset_index(
            drop=True
        )
        if self.process_limit:
            to_process = to_process.head(self.process_limit)

        logger.info(
            "LLM(GGUF): %d a processar (%d já ok, %d total)",
            len(to_process),
            len(processed_ids),
            len(keyframes_df),
        )

        for count, (_, row) in enumerate(to_process.iterrows(), start=1):
            try:
                raw = {}
                self._load_model()
                for field, (prompt, mx) in self.prompts.items():
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
                tags_path = checkpoint_path.parent / self.tags_filename
                tag_idx = self.build_tag_index(all_results)
                with open(tags_path, "w", encoding="utf-8") as _tf:
                    import json as _json
                    _json.dump(tag_idx, _tf, indent=2, ensure_ascii=False)
                logger.info("Checkpoint: %d/%d", count, len(to_process))

        return all_results

    @staticmethod
    def build_tag_index(results: list[dict]) -> dict:
        from collections import defaultdict

        idx: dict[str, list] = defaultdict(list)
        for rec in results:
            for tag in rec.get("tags", []):
                idx[tag].append(rec.get("scene_id"))
        return dict(sorted(idx.items(), key=lambda x: len(x[1]), reverse=True))

    def save(self, results, tag_index, output_dir):
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
        import json

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
