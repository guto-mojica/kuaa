"""Eval seed generation — produces a sample queries JSON for the grading UI.

Task 33 of the Mojica redesign ships a small, hand-crafted set of sample
queries so the eval grading UI (``/eval``) has something to render on a
fresh install. The JSON shape is the contract consumed by
``api.services.eval_service._load_queries`` (Task 30): a list of query
dicts with ``id``, ``text``, ``source``, ``lang``, ``k``,
``candidate_count``, ``latency_ms``, ``created_when`` and a ``results``
list whose entries carry the per-candidate fields the rows template reads
(``scene_id``, ``film_slug``, ``film_title``, ``year``, ``timecode``,
``description``, ``tags``, ``score``, ``keyframe_url``).

The candidate slates here are deliberately placeholders — five queries
with the same nine candidates each — because Month 3's curator-annotation
work will replace them with real /api/search results once the full
multi-film library is migrated. The shape, not the content, is what
matters for the UI to render without crashing.

Run ``kuaa eval seed --run <id> [--queries N] [--root PATH]`` to
write a queries file (default: ``data/eval/default.queries.json``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _mock_result(
    scene_id: int,
    film_slug: str,
    film_title: str,
    year: int,
    desc: str,
    tags: list[str],
    score: float,
    tc: str = "00:00:00",
) -> dict[str, Any]:
    """Build one candidate-result dict in the rows-template contract."""

    return {
        "scene_id": scene_id,
        "film_slug": film_slug,
        "film_title": film_title,
        "year": year,
        "timecode": tc,
        "description": desc,
        "tags": tags,
        "score": score,
        "keyframe_url": f"/media/library/{film_slug}/frames/scene_{scene_id:04d}.jpg",
    }


# Nine canonical candidates spanning the test fixture films. Reused across
# every seeded query — the M3 curator work replaces them with real search
# results once the multi-film library migration lands.
_PLACEHOLDER_RESULTS_OUTDOOR_DIALOG: list[dict[str, Any]] = [
    _mock_result(
        11,
        "jeca",
        "Jeca Tatu",
        1959,
        "Dois caboclos conversam à beira de uma cerca de pau-a-pique.",
        ["exterior", "duas-pessoas", "dialogo"],
        0.87,
        "00:21:58",
    ),
    _mock_result(
        3,
        "pagador",
        "O Pagador de Promessas",
        1962,
        "Zé do Burro e Rosa caminham em direção à igreja.",
        ["exterior", "rural", "caminhada"],
        0.85,
        "00:01:57",
    ),
    _mock_result(
        115,
        "jeca",
        "Jeca Tatu",
        1959,
        "Conversa entre vizinhos no terreiro da casa.",
        ["exterior", "duas-pessoas", "dia"],
        0.84,
        "00:22:15",
    ),
    _mock_result(
        6,
        "cangaceiro",
        "O Cangaceiro",
        1953,
        "Bando observa o horizonte do alto de uma chapada.",
        ["exterior", "grupo", "sertao"],
        0.82,
        "00:03:08",
    ),
    _mock_result(
        5,
        "aruanda",
        "Aruanda",
        1960,
        "Quilombolas conversam durante a colheita.",
        ["exterior", "labor", "documentario"],
        0.79,
        "00:02:51",
    ),
    _mock_result(
        4,
        "rio40",
        "Rio, 40 Graus",
        1955,
        "Vendedor de amendoim observa a praia ao longe.",
        ["exterior", "rua", "wagon"],
        0.78,
        "00:02:34",
    ),
    _mock_result(
        2,
        "jeca",
        "Jeca Tatu",
        1959,
        "Plano geral da casa do Jeca ao amanhecer.",
        ["exterior", "fence", "amanhecer"],
        0.76,
        "00:01:45",
    ),
    _mock_result(
        7,
        "pagador",
        "O Pagador de Promessas",
        1962,
        "Vista panorâmica da serra com a cruz ao fundo.",
        ["exterior", "paisagem", "mountain"],
        0.74,
        "00:03:18",
    ),
    _mock_result(
        48,
        "limite",
        "Limite",
        1931,
        "Interior de cabine com figura solitária recortada na janela.",
        ["interior", "baixa-luz", "experimental"],
        0.73,
        "00:14:08",
    ),
]

# All five seeded queries share the same nine placeholder candidates —
# the UI contract is "show a slate of nine cards on the centre pane",
# not "show *good* search results".  Real per-query slates land in
# Month 3 once /api/search is wired in.  Use the single list directly
# instead of maintaining four named copies that diverge semantically
# without adding information.
_PLACEHOLDER_RESULTS = _PLACEHOLDER_RESULTS_OUTDOOR_DIALOG


SAMPLE_QUERIES: list[dict[str, Any]] = [
    {
        "id": 1,
        "text": "duas pessoas conversando ao ar livre",
        "source": "manual",
        "lang": "pt",
        "k": 9,
        "candidate_count": 9,
        "latency_ms": 231,
        "created_when": "2026-04-12",
        "results": _PLACEHOLDER_RESULTS,
    },
    {
        "id": 2,
        "text": "cena noturna com fogo",
        "source": "manual",
        "lang": "pt",
        "k": 9,
        "candidate_count": 9,
        "latency_ms": 245,
        "created_when": "2026-04-12",
        "results": _PLACEHOLDER_RESULTS,
    },
    {
        "id": 3,
        "text": "a rider on horseback in a rural field",
        "source": "manual",
        "lang": "en",
        "k": 9,
        "candidate_count": 9,
        "latency_ms": 219,
        "created_when": "2026-04-12",
        "results": _PLACEHOLDER_RESULTS,
    },
    {
        "id": 4,
        "text": "interior baixa luz com figura solitária",
        "source": "manual",
        "lang": "pt",
        "k": 9,
        "candidate_count": 9,
        "latency_ms": 267,
        "created_when": "2026-04-12",
        "results": _PLACEHOLDER_RESULTS,
    },
    {
        "id": 5,
        "text": "cartela de título com escrita branca",
        "source": "manual",
        "lang": "pt",
        "k": 9,
        "candidate_count": 9,
        "latency_ms": 198,
        "created_when": "2026-04-12",
        "results": _PLACEHOLDER_RESULTS,
    },
]


_MAX_SEED_QUERIES = len(SAMPLE_QUERIES)


def write_seed(root: Path, run_id: str, count: int = 5) -> Path:
    """Write ``count`` sample queries to ``<root>/<run_id>.queries.json``.

    Returns the resolved output path. The parent directory is created on
    demand. ``count`` is clamped to ``[0, len(SAMPLE_QUERIES)]`` — passing
    a larger value silently writes every available sample rather than
    raising, because the CLI calls this with a user-supplied integer and
    we'd rather degrade gracefully than crash on ``--queries 99``.
    """

    root.mkdir(parents=True, exist_ok=True)
    clamped = max(0, min(count, _MAX_SEED_QUERIES))
    out_path = root / f"{run_id}.queries.json"
    queries = SAMPLE_QUERIES[:clamped]
    out_path.write_text(
        json.dumps(queries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out_path
