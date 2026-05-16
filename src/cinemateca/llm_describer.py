"""
cinemateca.llm_describer
~~~~~~~~~~~~~~~~~~~~~~~~
Geração automática de metadados descritivos para cenas usando
o modelo de linguagem visual Moondream 2 (vikhyatk/moondream2).

Baseado no Notebook 05 (05_descricao_llm.ipynb).

Estratégia de eficiência:
  - Prompt único combinado: 1 chamada ao decoder por frame (antes eram 6)
  - Encode da imagem feito UMA vez por frame
  - float16 em GPU/MPS, float32 em CPU
  - Processamento com checkpoint periódico (retomada automática)
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)


# ─── Constantes de modelo ────────────────────────────────────────────────────

# Moondream 2 usa SigLIP com entrada 378×378. Pré-redimensionar com PIL
# (rápido, bilinear) evita que o pyvips interno faça um resize multi-pass
# de alta qualidade (~20s/frame). PIL completa o mesmo trabalho em <0.1s.
_MOONDREAM_INPUT_SIZE = 378

# ─── Prompts individuais ─────────────────────────────────────────────────────
#
# Moondream 2 não segue instruções de formato JSON de forma confiável —
# retorna o template literal em vez de preencher os campos.
# Perguntas individuais curtas produzem respostas corretas e consistentes.
# max_new_tokens limita a geração para evitar respostas excessivamente longas.

PROMPTS: dict[str, tuple[str, int]] = {
    #                prompt                                          max_new_tokens
    "description":     ("Describe this film scene in one or two sentences. "
                        "Focus on the main subject, action, and setting.",      80),
    "location":        ("Is this scene indoors or outdoors? "
                        "Answer with one word: indoor or outdoor.",             10),
    "setting":         ("Describe the setting in 2-4 words. "
                        "Examples: urban street, rural field, farm, village.",  20),
    "time_of_day":     ("What time of day is this scene? "
                        "Answer with one word: day, night, or unknown.",        10),
    "people_and_action": ("How many people are visible and what are they doing? "
                          "Answer briefly, e.g.: 2 people talking.",            30),
    "objects":         ("List the most notable objects in this scene, "
                        "comma-separated. Maximum 6 items.",                    40),
}


LOCATION_MAP = {
    "indoor": "interior", "indoors": "interior", "inside": "interior",
    "interior": "interior", "internal": "interior",
    "outdoor": "exterior", "outdoors": "exterior", "outside": "exterior",
    "exterior": "exterior", "external": "exterior",
}

TIME_MAP = {
    "day": "dia", "daytime": "dia", "daylight": "dia",
    "night": "noite", "nighttime": "noite", "dark": "noite",
    "unknown": "desconhecido", "unclear": "desconhecido",
}


# ─── Parsing de respostas ─────────────────────────────────────────────────────

def _parse_num_people(text: str) -> int:
    """
    Extrai número de pessoas de uma resposta em linguagem natural.

    Retorna:
        int >= 0 : número conhecido
        -1       : múltiplos/vago ("several", "many")
    """
    text = text.lower()
    if any(w in text for w in ["no people", "nobody", "no person", "empty", "no one"]):
        return 0

    word_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "a person": 1, "a man": 1, "a woman": 1, "one person": 1,
    }
    for word, num in word_to_num.items():
        if word in text:
            return num

    match = re.search(r"\b(\d+)\b", text)
    if match:
        return int(match.group(1))

    if any(w in text for w in ["several", "many", "group", "crowd", "multiple"]):
        return -1

    return -1


def _parse_objects(text: str) -> list[str]:
    """Converte a string de objetos em lista normalizada (máx. 6 itens)."""
    if not text or text.startswith("ERROR"):
        return []
    parts = re.split(r"[,;.]+", text)
    stopwords = {"a", "an", "the", "some", "and", "with"}
    objects = []
    for p in parts:
        words = [w for w in p.strip().lower().split() if w not in stopwords]
        cleaned = " ".join(words)
        if cleaned:
            objects.append(cleaned)
    return objects[:6]


def _generate_tags(parsed: dict) -> list[str]:
    """Gera lista de tags kebab-case a partir dos campos parseados."""
    tags = set()

    if parsed.get("location") in ("interior", "exterior"):
        tags.add(parsed["location"])

    if parsed.get("time_of_day") in ("dia", "noite"):
        tags.add(parsed["time_of_day"])

    n = parsed.get("num_people", -1)
    if n == 0:
        tags.add("sem-pessoas")
    elif n == 1:
        tags.add("pessoa-unica")
    elif n == 2:
        tags.add("duas-pessoas")
    elif n >= 3:
        tags.add("multiplas-pessoas")

    for obj in parsed.get("objects", []):
        tags.add(obj.replace(" ", "-").lower())

    setting = parsed.get("setting", "").strip().lower()
    if setting:
        tags.add(setting.replace(" ", "-"))

    return sorted(tags)


# ─── Classe principal ─────────────────────────────────────────────────────────

class LLMDescriber:
    """
    Gera metadados descritivos para keyframes usando Moondream 2.

    Suporta:
        - Processamento individual ou em batch
        - Checkpoints periódicos para retomada após interrupções
        - Construção de índice invertido de tags

    Exemplo:
        describer = LLMDescriber(cfg, device)
        results = describer.describe_keyframes(keyframes_df)
        tag_index = describer.build_tag_index(results)
        describer.save(results, tag_index, cfg.paths.metadata_dir)
    """

    def __init__(self, cfg=None, device=None):
        self._model = None
        self._tokenizer = None
        self._device = device

        if cfg is not None:
            llm = cfg.llm
            self.model_id = llm.model_id
            self.revision = llm.revision
            self.checkpoint_interval = llm.checkpoint_interval
            self.process_limit = llm.process_limit
            self.descriptions_filename = llm.descriptions_filename
            self.tags_filename = llm.tags_filename
        else:
            self.model_id = "vikhyatk/moondream2"
            self.revision = "2025-01-09"
            self.checkpoint_interval = 25
            self.process_limit = None
            self.descriptions_filename = "scene_descriptions.json"
            self.tags_filename = "scene_tags.json"

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise RuntimeError(
                "transformers não instalado. Execute: pip install transformers"
            )

        import torch

        # float16 em GPU/MPS (2× mais rápido, metade da memória);
        # float32 em CPU (CPUs não têm SIMD nativo para float16)
        device_str = str(self._device) if self._device else "cpu"
        dtype = torch.float16 if device_str in ("cuda", "mps") else torch.float32

        logger.info(
            "Carregando Moondream 2 (%s) — dtype=%s device=%s — primeira execução baixa ~1.9GB...",
            self.revision, dtype, device_str,
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
        self._model = self._model.to(self._device)
        self._model.eval()

        logger.info("✓ Moondream 2 carregado em %.1fs", time.time() - t0)

    def _query_frame(self, image: Image.Image) -> dict:
        """Consulta o modelo com perguntas individuais, reutilizando o encoding."""
        self._load_model()
        image = image.resize(
            (_MOONDREAM_INPUT_SIZE, _MOONDREAM_INPUT_SIZE),
            Image.Resampling.BILINEAR,
        )
        enc = self._model.encode_image(image)
        raw = {}
        for field, (prompt, max_tokens) in PROMPTS.items():
            try:
                raw[field] = self._model.answer_question(
                    enc, prompt, self._tokenizer, max_new_tokens=max_tokens
                ).strip()
            except Exception as e:
                raw[field] = f"ERROR: {e}"
        return raw

    def _build_metadata(self, row: pd.Series, raw: dict) -> dict:
        """Monta o dict final de metadados combinando dados do catálogo + respostas LLM."""
        loc_raw = raw.get("location", "").lower().strip()
        location = LOCATION_MAP.get(loc_raw, "desconhecido")

        time_raw = raw.get("time_of_day", "").lower().strip()
        time_of_day = TIME_MAP.get(time_raw, "desconhecido")

        num_people = _parse_num_people(raw.get("people_and_action", ""))
        objects = _parse_objects(raw.get("objects", ""))
        setting = raw.get("setting", "").strip().lower()

        parsed = {
            "location": location,
            "time_of_day": time_of_day,
            "num_people": num_people,
            "objects": objects,
            "setting": setting,
        }
        tags = _generate_tags(parsed)

        scene_meta = {
            "scene_id": int(row.get("scene_id", -1)),
            "keyframe_id": str(row.get("keyframe_id", "")),
            "keyframe_path": str(row.get("filepath", "")),
            "start_time_s": float(row["start_time_s"]) if "start_time_s" in row.index else None,
            "end_time_s": float(row["end_time_s"]) if "end_time_s" in row.index else None,
            "duration_s": float(row["duration_s"]) if "duration_s" in row.index else None,
        }

        llm_meta = {
            "description": raw.get("description", "").strip(),
            "location": location,
            "setting": setting,
            "time_of_day": time_of_day,
            "num_people": num_people,
            "people_action": raw.get("people_and_action", "").strip(),
            "objects": objects,
            "tags": tags,
            "_raw_responses": raw,  # útil para depuração e auditoria
        }

        return {**scene_meta, **llm_meta}

    def describe_keyframes(
        self,
        keyframes_df: pd.DataFrame,
        existing_results: list[dict] | None = None,
        checkpoint_path: Path | None = None,
    ) -> list[dict]:
        """
        Processa todos os keyframes, gerando metadados descritivos.

        Suporta retomada: se existing_results for fornecido, pula os
        scene_ids já presentes.

        Args:
            keyframes_df:     DataFrame com colunas filepath, scene_id, etc.
            existing_results: Resultados de execução anterior (para retomada).
            checkpoint_path:  Onde salvar checkpoints periódicos.

        Returns:
            Lista de dicts com metadados completos.
        """
        all_results = list(existing_results or [])
        processed_ids = {r["scene_id"] for r in all_results}

        to_process = keyframes_df[
            ~keyframes_df["scene_id"].isin(processed_ids)
        ].reset_index(drop=True)

        if self.process_limit:
            to_process = to_process.head(self.process_limit)

        logger.info(
            "LLM: %d frames a processar (%d já processados, %d total)",
            len(to_process),
            len(processed_ids),
            len(keyframes_df),
        )

        errors = []
        times = []

        for i, row in to_process.iterrows():
            t0 = time.time()
            try:
                img = Image.open(row["filepath"]).convert("RGB")
                raw = self._query_frame(img)
                meta = self._build_metadata(row, raw)
                all_results.append(meta)
            except Exception as e:
                err = {
                    "scene_id": int(row.get("scene_id", -1)),
                    "keyframe_path": str(row["filepath"]),
                    "error": str(e),
                    "tags": [],
                    "objects": [],
                }
                errors.append(err)
                all_results.append(err)
                logger.error("Erro ao processar cena %s: %s", row.get("scene_id"), e)

            elapsed = time.time() - t0
            times.append(elapsed)

            # Checkpoint periódico
            count = i - to_process.index[0] + 1 if len(to_process) > 0 else 0
            if checkpoint_path and count % self.checkpoint_interval == 0:
                self._save_json(all_results, checkpoint_path)
                remaining = len(to_process) - count
                eta = np.mean(times[-self.checkpoint_interval:]) * remaining / 60
                logger.info(
                    "Checkpoint: %d/%d | ETA: %.0f min | Erros: %d",
                    count, len(to_process), eta, len(errors),
                )

        if times:
            logger.info(
                "✓ LLM concluído: %d cenas | média %.1fs/frame | erros: %d",
                len(all_results),
                np.mean(times),
                len(errors),
            )
        return all_results

    @staticmethod
    def build_tag_index(results: list[dict]) -> dict:
        """
        Constrói índice invertido: tag → [scene_ids].

        Útil para filtrar cenas rapidamente antes da busca semântica.
        """
        tag_index: dict[str, list] = defaultdict(list)
        for record in results:
            sid = record.get("scene_id")
            for tag in record.get("tags", []):
                tag_index[tag].append(sid)
        # Ordenar por frequência (mais comuns primeiro)
        return dict(
            sorted(tag_index.items(), key=lambda x: len(x[1]), reverse=True)
        )

    def save(
        self,
        results: list[dict],
        tag_index: dict,
        output_dir: str | Path,
    ) -> tuple[Path, Path]:
        """
        Salva descrições e índice de tags em JSON.

        Returns:
            Tupla (descriptions_path, tags_path).
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        desc_path = out / self.descriptions_filename
        tags_path = out / self.tags_filename

        self._save_json(results, desc_path)
        self._save_json(tag_index, tags_path)

        logger.info("✓ Descrições salvas: %s (%d cenas)", desc_path, len(results))
        logger.info("✓ Tags salvas: %s (%d tags únicas)", tags_path, len(tag_index))

        return desc_path, tags_path

    @staticmethod
    def load(descriptions_path: str | Path, tags_path: str | Path) -> tuple[list[dict], dict]:
        """Carrega descrições e índice de tags do disco."""
        with open(descriptions_path, encoding="utf-8") as f:
            results = json.load(f)
        with open(tags_path, encoding="utf-8") as f:
            tag_index = json.load(f)
        logger.info(
            "✓ LLM carregado: %d cenas, %d tags", len(results), len(tag_index)
        )
        return results, tag_index

    @staticmethod
    def _save_json(data, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
