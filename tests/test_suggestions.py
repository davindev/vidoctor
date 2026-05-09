"""suggestions.py — finding_refs 검증 등 순수 함수 단위 테스트."""

from __future__ import annotations

from vidoctor.suggestions import (
    _RefValidationStats,
    _SuggestionItem,
    _validate_refs,
)


def _item(text: str, refs: list[str]) -> _SuggestionItem:
    return _SuggestionItem(text=text, finding_refs=refs)


def test_validate_refs_keeps_only_valid_refs():
    items = [_item("a", ["filler:0", "filler:99", "gaze:0"])]
    cleaned, stats = _validate_refs(items, valid_refs={"filler:0", "cps:1"})
    assert cleaned[0].finding_refs == ["filler:0"]
    assert stats == _RefValidationStats(
        input_suggestions=1,
        kept_suggestions=1,
        invalid_refs_removed=2,
        suggestions_dropped=0,
    )


def test_validate_refs_drops_suggestion_with_no_valid_refs():
    items = [
        _item("legit", ["filler:0"]),
        _item("hallucinated", ["filler:99", "gaze:42"]),
    ]
    cleaned, stats = _validate_refs(items, valid_refs={"filler:0"})
    assert [s.text for s in cleaned] == ["legit"]
    assert stats.kept_suggestions == 1
    assert stats.suggestions_dropped == 1
    assert stats.invalid_refs_removed == 2


def test_validate_refs_empty_input_yields_empty_stats():
    cleaned, stats = _validate_refs([], valid_refs={"filler:0"})
    assert cleaned == []
    assert stats == _RefValidationStats(
        input_suggestions=0,
        kept_suggestions=0,
        invalid_refs_removed=0,
        suggestions_dropped=0,
    )


def test_validate_refs_all_valid_yields_zero_invalid_count():
    items = [_item("a", ["filler:0", "cps:1"])]
    cleaned, stats = _validate_refs(items, valid_refs={"filler:0", "cps:1"})
    assert cleaned[0].finding_refs == ["filler:0", "cps:1"]
    assert stats.invalid_refs_removed == 0
    assert stats.suggestions_dropped == 0


def test_validate_refs_preserves_text_and_other_fields():
    items = [_item("본문", ["filler:0", "filler:99"])]
    cleaned, _ = _validate_refs(items, valid_refs={"filler:0"})
    assert cleaned[0].text == "본문"


def test_validate_refs_does_not_mutate_input():
    original = _item("a", ["filler:0", "filler:99"])
    items = [original]
    _validate_refs(items, valid_refs={"filler:0"})
    assert original.finding_refs == ["filler:0", "filler:99"]
