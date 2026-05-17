"""
cinemateca.models.describer.gguf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Keyless, offline Moondream 2 scene describer via llama-cpp-python.
Loads the 2025-01-09 GGUF pair (text model + mmproj) from the
vikhyatk/moondream2 repo at that revision. No transformers, no API key.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from cinemateca.models.describer._common import (
    PROMPTS,
    build_metadata,
)

logger = logging.getLogger(__name__)

_REPO = "vikhyatk/moondream2"
_REV = "2025-01-09"
_TEXT_GGUF = "moondream2-text-model-f16.gguf"
_MMPROJ_GGUF = "moondream2-mmproj-f16.gguf"


class MoondreamGGUFDescriber:
    """SceneDescriber backed by Moondream 2 GGUF + llama-cpp-python."""

    def __init__(self, cfg=None, device=None):
        self._llm = None
        if cfg is not None and hasattr(cfg, "llm"):
            self.checkpoint_interval = cfg.llm.checkpoint_interval
            self.process_limit = cfg.llm.process_limit
            self.descriptions_filename = cfg.llm.descriptions_filename
            self.tags_filename = cfg.llm.tags_filename
        else:
            self.checkpoint_interval = 25
            self.process_limit = None
            self.descriptions_filename = "scene_descriptions.json"
            self.tags_filename = "scene_tags.json"

    def _load_model(self):
        if self._llm is not None:
            return
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
            verbose=False,
        )
        logger.info("✓ Moondream GGUF carregado em %.1fs", time.time() - t0)

    def _answer(self, image_path, prompt: str, max_tokens: int) -> str:
        """One image+prompt -> stripped model text. Re-embeds per call."""
        uri = Path(image_path).resolve().as_uri()
        resp = self._llm.create_chat_completion(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": uri}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=max_tokens,
        )
        return resp["choices"][0]["message"]["content"].strip()

    def describe(self, image_path) -> dict:
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
        # RESUME-BUG FIX: error rows are NOT counted as processed — they are
        # dropped so reprocessing can produce a good result (mirror pipeline.py:332).
        existing = list(existing_results or [])
        processed_ids = {r["scene_id"] for r in existing if "error" not in r}
        all_results = [r for r in existing if "error" not in r]
        to_process = keyframes_df[
            ~keyframes_df["scene_id"].isin(processed_ids)
        ].reset_index(drop=True)
        if self.process_limit:
            to_process = to_process.head(self.process_limit)

        logger.info(
            "LLM(GGUF): %d a processar (%d já ok, %d total)",
            len(to_process), len(processed_ids), len(keyframes_df),
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
            except Exception as e:  # noqa: BLE001 - whole-frame failure
                all_results.append({
                    "scene_id": int(row.get("scene_id", -1)),
                    "keyframe_path": str(row["filepath"]),
                    "error": str(e),
                    "tags": [],
                    "objects": [],
                })
                logger.error("Erro cena %s: %s", row.get("scene_id"), e)

            if checkpoint_path and count % self.checkpoint_interval == 0:
                self._save_json(all_results, checkpoint_path)
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
