"""Filler 검출 테스트 — 사전 매칭 + 반복 어휘 묶음."""

import pytest

from tests._helpers import w
from vidoctor.audio.filler import detect_filler_events


@pytest.mark.parametrize("text", ["음", "어", "그", "이제"])
def test_filler_dictionary_word_detected(text: str):
    events = detect_filler_events([w(text, 0.0, 0.3)])
    assert len(events) == 1
    assert events[0].text == text


def test_punctuation_normalized():
    events = detect_filler_events([w("어,", 0.0, 0.2)])
    assert len(events) == 1


def test_normal_word_not_detected():
    events = detect_filler_events([w("안녕하세요", 0.0, 0.5)])
    assert events == []


def test_logical_connective_not_filler():
    # "그러니까/그래서"는 강의에서 논리 연결사로 정상 사용 — false positive 차단 의도.
    # filler.py FILLERS 사전에서 의도적 미포함이므로 detection 0건이어야 한다.
    events = detect_filler_events([w("그러니까", 0.0, 0.4)])
    assert events == []


def test_repetition_detected():
    # "그 그" — 0.1s gap < 0.5s
    words = [w("그", 0.0, 0.2), w("그", 0.3, 0.5)]
    events = detect_filler_events(words)
    # 2 Tier 1 단일 + 1 반복 = 3
    repetitions = [e for e in events if " " in e.text]
    assert len(repetitions) == 1


def test_repetition_too_far_apart_not_merged():
    # 1.0s gap ≥ 0.5s → 반복 아님
    words = [w("그", 0.0, 0.2), w("그", 1.5, 1.7)]
    events = detect_filler_events(words)
    repetitions = [e for e in events if " " in e.text]
    assert len(repetitions) == 0


def test_empty_input():
    assert detect_filler_events([]) == []


def test_mixed_natural_with_fillers():
    words = [
        w("안녕하세요", 0.0, 0.5),
        w("음", 0.6, 0.7),
        w("오늘은", 1.0, 1.3),
        w("그", 1.4, 1.5),
        w("타입스크립트를", 1.7, 2.3),
    ]
    events = detect_filler_events(words)
    # "음", "그" 두 개 검출
    detected_texts = [e.text for e in events]
    assert "음" in detected_texts
    assert "그" in detected_texts
    assert "안녕하세요" not in detected_texts
