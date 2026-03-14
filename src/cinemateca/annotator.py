"""
cinemateca.annotator
~~~~~~~~~~~~~~~~~~~~
Anotação manual de cenas: leitura, escrita e mesclagem com tags automáticas.

As anotações manuais são salvas em 'manual_annotations.json' no mesmo
diretório dos metadados. O formato é um dict simples:

    { "351": ["tag1", "tag2"], "352": ["rural", "exterior"] }

A função merge_tag_index() combina essas anotações com o índice de tags
gerado pelo LLM (scene_tags.json) para que a busca e o catálogo
reflitam as duas fontes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

FILENAME = "manual_annotations.json"


def load(metadata_dir: str | Path) -> Dict[str, List[str]]:
    """
    Carrega anotações manuais do disco.

    Returns:
        Dict de {scene_id (str): [tags]}. Vazio se o arquivo não existir.
    """
    path = Path(metadata_dir) / FILENAME
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    logger.debug("Anotações manuais carregadas: %d cenas anotadas", len(data))
    return data


def save(metadata_dir: str | Path, annotations: Dict[str, List[str]]) -> Path:
    """
    Persiste o dict de anotações no disco.

    Args:
        metadata_dir:  Diretório de metadados do projeto.
        annotations:   Dict {scene_id (str): [tags]}.

    Returns:
        Path do arquivo salvo.
    """
    path = Path(metadata_dir) / FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    logger.info("✓ Anotações manuais salvas: %s (%d cenas)", path, len(annotations))
    return path


def merge_tag_index(
    llm_tag_index: Dict[str, List[str]] | None,
    annotations: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """
    Mescla o índice de tags do LLM com as anotações manuais.

    O índice de tags do LLM tem o formato invertido:
        { "tag": ["scene_id1", "scene_id2"] }

    As anotações manuais têm o formato:
        { "scene_id": ["tag1", "tag2"] }

    Returns:
        Índice mesclado no formato { "tag": [scene_ids] }.
    """
    merged: Dict[str, List[str]] = {}

    # Copiar tags do LLM
    if llm_tag_index:
        for tag, ids in llm_tag_index.items():
            merged[tag] = list(ids)

    # Adicionar tags manuais
    for scene_id, tags in annotations.items():
        for tag in tags:
            tag = tag.strip().lower().replace(" ", "-")
            if not tag:
                continue
            if tag not in merged:
                merged[tag] = []
            if scene_id not in merged[tag]:
                merged[tag].append(scene_id)

    return merged
