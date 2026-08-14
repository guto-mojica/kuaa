"""
kuaa.models.describer.transformers_hf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Default scene describer: Moondream 2 via Hugging Face transformers.

Loads ``vikhyatk/moondream2`` at the configured revision (default
``2025-01-09``) with ``trust_remote_code``. GPU acceleration comes from
the installed PyTorch wheel (CUDA / Apple MPS) — no source build. The
2025-01-09 remote code is transformers-4 only; the dependency is pinned
``transformers>=4.44,<5`` in pyproject (verified 2026-05-18: tf5 hard-fails
for every moondream2 revision).

Parsing, tagging and resume behaviour are shared verbatim with every other
describer backend via :mod:`kuaa.models.describer._common`, so
switching backends does not change metadata semantics.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any, cast

import pandas as pd

from kuaa.config import Settings
from kuaa.errors import ModelError
from kuaa.models.base import SceneDescriptionRecord
from kuaa.models.describer._common import (
    DEFAULT_MAX_CONSECUTIVE_DEGENERATE,
    DEGENERACY_ABORT_HINT,
    PROMPTS,
    build_metadata,
    degenerate_fields,
)
from kuaa.models.describer.domain_prompts import prompts_from_config
from kuaa.models.manifest import ModelCard, get_card

logger = logging.getLogger(__name__)

# Moondream 2 uses a SigLIP encoder with 378x378 input. Pre-resizing with
# PIL (fast, bilinear) avoids the model's internal high-quality multi-pass
# resize (~20s/frame) for the same effective input.
_INPUT_SIZE = 378


class MoondreamTransformersDescriber:
    """SceneDescriber backed by Moondream 2 via HF transformers."""

    #: Provenance for this backend (manifest single source of truth, C10/F6).
    CARD: ModelCard = get_card("moondream_transformers")

    def __init__(self, cfg: Settings | None = None, device=None):
        # Lazy-loaded model + tokenizer (populated by ``_load_model``); typed
        # ``Any`` so the lazy-``None`` initial value doesn't poison call sites.
        self._model: Any = None
        self._tokenizer: Any = None
        self._enc_cache: tuple[str, object] | None = None
        self._device = device
        # Whether the loaded revision honours a per-call token cap. Probed at
        # load time and demoted on first failure; see ``_answer``.
        self._token_cap_supported = True
        if cfg is not None and getattr(cfg, "llm", None) is not None:
            self.model_id = cfg.llm.model_id
            self.revision = cfg.llm.revision
            self.checkpoint_interval = cfg.llm.checkpoint_interval
            self.process_limit = cfg.llm.process_limit
            self.descriptions_filename = cfg.llm.descriptions_filename
            self.tags_filename = cfg.llm.tags_filename
            self.max_consecutive_degenerate = getattr(
                cfg.llm,
                "max_consecutive_degenerate",
                DEFAULT_MAX_CONSECUTIVE_DEGENERATE,
            )
            self.prompts = prompts_from_config(cfg)
        else:
            self.model_id = "vikhyatk/moondream2"
            self.revision = "2025-01-09"
            self.checkpoint_interval = 25
            self.process_limit = None
            self.descriptions_filename = "scene_descriptions.json"
            self.tags_filename = "scene_tags.json"
            self.max_consecutive_degenerate = DEFAULT_MAX_CONSECUTIVE_DEGENERATE
            self.prompts = dict(PROMPTS)

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
            device_map=self._device if self._device is not None else "cpu",
        )
        self._model.eval()
        self._token_cap_supported = hasattr(self._model, "query")
        if not self._token_cap_supported:
            logger.warning(
                "Revisão %s não expõe query(settings=...) — o teto de tokens por "
                "prompt NÃO será aplicado e cada resposta pode ir até o limite "
                "interno do modelo (512). Prefira a revisão 2025-01-09.",
                self.revision,
            )
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

        # Drop the previous frame's encoding *before* allocating the next.
        # An EncodedImage owns a static KV cache of
        # n_layers(24) x 2 x n_heads(32) x max_context(2048) x head_dim(64)
        # in fp16 = 384 MiB. Assigning the new tuple first would hold both
        # alive across the allocation and double the peak for no reason.
        self._enc_cache = None
        img = (
            Image.open(image_path)
            .convert("RGB")
            .resize((_INPUT_SIZE, _INPUT_SIZE), Image.Resampling.BILINEAR)
        )
        enc = self._model.encode_image(img)
        self._enc_cache = (key, enc)
        return enc

    def release(self) -> None:
        """Drop the model and hand its device memory back to the allocator.

        Necessary because the accelerator caching allocators (MPS and CUDA
        alike) keep freed blocks in their own pool: dropping the last Python
        reference to a 3.7 GB fp16 Moondream returns nothing to the OS on its
        own. In a long-lived ``kuaa serve`` process that describes several
        films, each film's describer stacked on the previous one's retained
        blocks until the GPU had nothing left — the 2026-08-14 MPS OOM.

        Idempotent, and safe to call on a describer that never loaded.
        """
        had_model = self._model is not None
        self._enc_cache = None
        self._model = None
        self._tokenizer = None
        if not had_model:
            return
        import gc

        gc.collect()
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - reclaiming memory must never fail a run
            logger.debug("empty_cache falhou; seguindo", exc_info=True)

    def _answer(self, image_path, prompt: str, max_tokens: int) -> str:
        """One image+prompt -> stripped model text (encoder cached per frame).

        Routes through ``query(settings={"max_tokens": N})`` because
        ``answer_question`` **silently ignores** ``max_new_tokens`` on the
        2025-01-09 remote code — it forwards to ``query`` with no settings,
        so every prompt ran to ``DEFAULT_MAX_TOKENS`` (512). The sibling
        ``generate`` says so in its own docstring: "tokenizer, max_new_takens,
        and kwargs are ignored."

        That made the whole ``PROMPTS`` budget (190 tokens/scene) dead config
        and let a single runaway generation cost 16x its cap in time and KV
        cache. ``answer_question`` stays as the fallback for revisions that
        predate ``query`` — but loudly, and only once.
        """
        enc = self._encode(image_path)
        if self._token_cap_supported:
            # Resolved by name rather than called directly so an AttributeError
            # raised *inside* query() is never mistaken for "query is missing".
            query = getattr(self._model, "query", None)
            if query is None:
                self._demote_token_cap("revisão não expõe query()")
            else:
                try:
                    answer = query(enc, prompt, settings={"max_tokens": max_tokens})
                    return str(answer["answer"]).strip()
                except (NotImplementedError, TypeError, KeyError) as e:
                    self._demote_token_cap(repr(e))
        return self._model.answer_question(
            enc, prompt, self._tokenizer, max_new_tokens=max_tokens
        ).strip()

    def _demote_token_cap(self, reason: str) -> None:
        """Fall back to the uncapped path — once, and loudly."""
        self._token_cap_supported = False
        logger.warning(
            "query(settings=...) indisponível (%s) — caindo para answer_question, "
            "que IGNORA o teto de tokens nesta revisão; respostas podem ir até 512 "
            "tokens. Prefira a revisão 2025-01-09.",
            reason,
        )

    def describe(self, image_path) -> dict:
        """Return full metadata dict for a single keyframe."""
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
        existing_results: list[SceneDescriptionRecord] | None = None,
        checkpoint_path: Path | None = None,
    ) -> list[SceneDescriptionRecord]:
        """Process all rows; resume via existing_results.

        RESUME-BUG FIX (mirrors gguf.py): error rows are NOT counted as
        processed — they are dropped so reprocessing can produce a good
        result; good rows are preserved verbatim, not rebuilt.

        Aborts with :class:`~kuaa.errors.ModelError` after
        ``max_consecutive_degenerate`` scenes come back as repetition loops,
        checkpointing first so the good rows survive and resume picks up from
        them. Without this the 2026-08-14 run spent 6.5 hours producing one
        usable scene out of twenty before the GPU failed hard.

        Always releases the model on the way out — see :meth:`release`.
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

        consecutive_degenerate = 0
        try:
            for count, (_, row) in enumerate(to_process.iterrows(), start=1):
                scene_id = row.get("scene_id", -1)
                try:
                    raw = {}
                    self._load_model()
                    for field, (prompt, mx) in self.prompts.items():
                        try:
                            raw[field] = self._answer(row["filepath"], prompt, mx)
                        except Exception as e:  # noqa: BLE001
                            raw[field] = f"ERROR: {e}"
                    # Read degeneracy from the raw answers rather than sniffing
                    # the error string build_metadata writes: the breaker and
                    # the record must agree on what happened, by construction.
                    bad_fields = degenerate_fields(raw)
                    meta = build_metadata(row, raw)
                    all_results.append(cast(SceneDescriptionRecord, meta))
                    if bad_fields:
                        consecutive_degenerate += 1
                        logger.error(
                            "cena %s [%d/%d]: saída degenerada em %s — cena "
                            "descartada (%d consecutiva[s])",
                            meta.get("scene_id"),
                            count,
                            len(to_process),
                            ", ".join(bad_fields),
                            consecutive_degenerate,
                        )
                    else:
                        consecutive_degenerate = 0
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
                            "scene_id": int(scene_id),
                            "keyframe_path": str(row["filepath"]),
                            "error": str(e),
                            "tags": [],
                            "objects": [],
                        }
                    )
                    logger.error("Erro cena %s: %s", scene_id, e)

                # Circuit breaker before the periodic checkpoint so the abort
                # path owns the final write (0 disables the breaker).
                if (
                    self.max_consecutive_degenerate > 0
                    and consecutive_degenerate >= self.max_consecutive_degenerate
                ):
                    if checkpoint_path:
                        self._save_json(all_results, checkpoint_path)
                        logger.info("Checkpoint antes de abortar: %d/%d", count, len(to_process))
                    raise ModelError(
                        f"{consecutive_degenerate} cenas consecutivas com saída "
                        f"degenerada (última: cena {scene_id}, {count}/{len(to_process)}). "
                        f"{DEGENERACY_ABORT_HINT}"
                    )

                if checkpoint_path and count % self.checkpoint_interval == 0:
                    self._save_json(all_results, checkpoint_path)
                    logger.info("Checkpoint: %d/%d", count, len(to_process))
        finally:
            # A film's worth of weights must not outlive the step that needed
            # them — see release(). Runs on the abort path too.
            self.release()

        return all_results

    @staticmethod
    def build_tag_index(results: list[SceneDescriptionRecord]) -> dict[str, list[str]]:
        """Build a tag → [scene_id] index sorted by frequency descending.

        Scene IDs are stored as strings so the index type is homogeneous
        (``dict[str, list[str]]``) and consumers can do membership tests
        without worrying about int/str divergence.
        """
        from collections import defaultdict

        idx: dict[str, list[str]] = defaultdict(list)
        for rec in results:
            sid = rec.get("scene_id")
            if sid is not None:
                for tag in rec.get("tags", []):
                    idx[tag].append(str(sid))
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
