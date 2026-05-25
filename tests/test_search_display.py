"""Unit tests for cinemateca.search.display — degenerate-tag filtering."""

from __future__ import annotations

from cinemateca.search.display import filter_degenerate_tags, is_degenerate_tag


def test_keeps_curated_tags():
    assert not is_degenerate_tag("outdoor")
    assert not is_degenerate_tag("rural-field")
    assert not is_degenerate_tag("man-on-horse")


def test_drops_empty():
    assert is_degenerate_tag("")


def test_drops_pure_digit():
    assert is_degenerate_tag("42")


def test_drops_long_caption():
    assert is_degenerate_tag("a-long-string-that-is-clearly-a-sentence")


def test_drops_internal_period():
    assert is_degenerate_tag("a-baby.in-a-basket")


def test_keeps_trailing_period_when_not_article_led():
    assert not is_degenerate_tag("farm.")


def test_drops_article_led_period():
    assert is_degenerate_tag("a-baby-in-a-basket.")


def test_drops_repeated_token():
    assert is_degenerate_tag("gate-gate-gate")


def test_drops_excess_hyphens():
    assert is_degenerate_tag("a-b-c-d-e")


def test_drops_digit_led():
    assert is_degenerate_tag("1-cow")


def test_drops_numeric_suffix():
    assert is_degenerate_tag("man-in-hat-2")


def test_filter_drops_degenerates_and_preserves_curated():
    tags = ["outdoor", "42", "rural-field", "gate-gate-gate", "farm."]
    assert filter_degenerate_tags(tags) == ["outdoor", "rural-field", "farm."]
