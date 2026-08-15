"""Portuguese retrieval must not silently die again.

Three failures stacked to make PT search return nothing while the English
equivalent worked, and no test caught any of them:

1. The BM25 corpus is English (English describer prompts), so PT content
   queries scored literally zero.
2. ``MetadataScorer`` bailed out on any query over 4 tokens, disabling the
   0.65-weighted fusion leg on 13 of the project's own 15 eval queries.
3. The PT tokenizer option raised ``ModuleNotFoundError`` because ``nltk``
   is not a dependency, and two of the three BM25 load sites dropped the
   config knob anyway.

The unit tests here run everywhere. The corpus-level test is skipped when
the library is not indexed, so it must be run locally to mean anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kuaa.errors import is_error_response
from kuaa.retrieval.bilingual import en_forms, expand_text, pt_forms
from kuaa.retrieval.bm25 import BM25Index
from kuaa.retrieval.tokenize import get_tokenizer
from kuaa.search._aggregate.scorers import MetadataScorer, build_query_terms

_LIBRARY = Path("data/library")
_SNAPSHOT_FILM = "jeca_tatu_1959"


# ── the lexicon ──────────────────────────────────────────────────────────────


def test_lexicon_round_trips_core_archive_terms() -> None:
    for english, portuguese in [("horse", "cavalo"), ("house", "casa"), ("hat", "chapéu")]:
        assert portuguese in pt_forms(english)
        assert english in en_forms(portuguese)


def test_en_forms_accepts_unaccented_spelling() -> None:
    """A curator may type ``chapeu`` or ``chapéu``; both must resolve."""
    assert en_forms("chapeu") == en_forms("chapéu")
    assert "hat" in en_forms("chapeu")


def test_expand_text_is_additive_and_stable() -> None:
    out = expand_text("a man rides a horse")
    assert "homem" in out and "cavalo" in out
    # Deterministic, so a rebuilt index is byte-identical for equal input.
    assert out == expand_text("a man rides a horse")


def test_expand_text_ignores_unknown_english() -> None:
    assert expand_text("frisbee snowboard") == ""


# ── the tokenizer ────────────────────────────────────────────────────────────


def test_multilingual_tokenizer_needs_no_optional_dependency() -> None:
    """Regression: this used to raise ModuleNotFoundError on a missing nltk.

    The exception escaped as a 500, because the caller in
    ``kuaa.search.aggregate`` catches only FileNotFoundError/OSError/ValueError.
    """
    assert get_tokenizer("multilingual").tokenize("o homem no cavalo")


def test_tokenizer_bridges_the_accent_the_pipeline_drops() -> None:
    """The describer writes ``pessoa-unica``; a curator types ``única``."""
    tok = get_tokenizer("multilingual")
    assert tok.tokenize("única") == tok.tokenize("unica")


# ── the metadata leg ─────────────────────────────────────────────────────────


def test_metadata_leg_survives_long_natural_language_queries() -> None:
    """The 4-token bail-out killed this leg on almost every real query."""
    long_query = "homem com chapéu fumando dentro de casa em cenário rural"
    assert build_query_terms(long_query)


def test_metadata_leg_scores_partial_coverage() -> None:
    """All-or-nothing matching meant a 7-word query scored zero on its own scene."""
    scorer = MetadataScorer()
    scores = scorer.score(
        query="man in a hat walking down a dirt road",
        descriptions=[
            {
                "scene_id": 1,
                "description": "A man in a straw hat walks down a dirt road past a fence.",
            }
        ],
        tag_index={},
        visual_rows=[],
    )
    assert scores.get(1, 0.0) > 0.0


def test_metadata_leg_still_rejects_incidental_single_word_matches() -> None:
    """Partial credit must not turn one shared word into a match."""
    scorer = MetadataScorer()
    scores = scorer.score(
        query="mulher carregando criança em cenário rural",
        descriptions=[{"scene_id": 1, "description": "A wooden cart stands in a rural field."}],
        tag_index={},
        visual_rows=[],
    )
    assert scores == {}


def test_portuguese_and_english_score_the_same_detector_class() -> None:
    scorer = MetadataScorer()
    rows = [
        {
            "scene_id": 1,
            "object_detection": {"objects": [{"class": "horse"}], "class_counts": {"horse": 1}},
        }
    ]
    kwargs: dict = {"descriptions": [], "tag_index": {}, "visual_rows": rows}
    assert scorer.score(query="cavalo", **kwargs) == scorer.score(query="horse", **kwargs)


# ── failure text must never become content ───────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "Error: Passed CPU tensor to MPS op",  # the one that actually shipped
        "ERROR: model unavailable",
        "erro: falha ao carregar",
        "Traceback (most recent call last)",
    ],
)
def test_captured_failures_are_recognised(text: str) -> None:
    assert is_error_response(text)


@pytest.mark.parametrize("text", ["urban street", "a man with an error-proof hat", "rural field"])
def test_real_captions_are_not_mistaken_for_failures(text: str) -> None:
    assert not is_error_response(text)


def test_error_shaped_tags_never_enter_the_corpus() -> None:
    # Five documents: BM25 IDF goes to zero for a term carried by half the
    # corpus, so a two-doc fixture cannot demonstrate retrieval either way.
    index = BM25Index.build(
        descriptions=[
            {"scene_id": 1, "description": "a man on a horse"},
            {"scene_id": 2, "description": "an empty room"},
            {"scene_id": 3, "description": "a woman at a table"},
            {"scene_id": 4, "description": "a dirt road at night"},
            {"scene_id": 5, "description": "two children on a porch"},
        ],
        tag_index={
            "error:-passed-cpu-tensor-to-mps-op": [1, 2, 3, 4, 5],
            "horse": [1],
        },
    )
    assert index.query("error", 5) == []
    assert [sid for sid, _ in index.query("horse", 5)] == [1]


# ── against the real corpus ──────────────────────────────────────────────────


def _indexed() -> bool:
    return (_LIBRARY / _SNAPSHOT_FILM / "metadata" / "scene_descriptions.json").is_file()


@pytest.mark.skipif(not _indexed(), reason=f"{_SNAPSHOT_FILM} not indexed on this machine")
@pytest.mark.parametrize(
    "pt_query",
    [
        "homem a cavalo",
        "mulher de vestido",
        "cavalo",
        "casa",
        "dois homens conversando",
        "chapéu de palha",
    ],
)
def test_portuguese_queries_retrieve_from_the_real_corpus(pt_query: str) -> None:
    """Every one of these returned zero hits before the bilingual expansion."""
    md = _LIBRARY / _SNAPSHOT_FILM / "metadata"
    descriptions = json.loads((md / "scene_descriptions.json").read_text())
    tag_index = json.loads((md / "scene_tags.json").read_text())
    index = BM25Index.build(
        descriptions=descriptions,
        tag_index=tag_index,
        tokenizer=get_tokenizer("multilingual"),
        bilingual=True,
    )
    assert index.query(pt_query, 10), f"{pt_query!r} retrieved nothing"
