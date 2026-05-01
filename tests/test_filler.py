from tests._helpers import _w
from vidoctor.audio.filler import detect_filler_events


def test_tier1_filler_detected():
    events = detect_filler_events([_w("음", 1.0, 1.2)])
    assert len(events) == 1
    assert events[0].text == "음"
    assert events[0].severity == "low"


def test_tier1_long_vowel_severity_mid():
    # duration 0.5s ≥ 0.4s → 모음 늘임 → severity mid
    events = detect_filler_events([_w("어", 0.0, 0.5)])
    assert events[0].severity == "mid"


def test_tier2_filler_severity_low():
    events = detect_filler_events([_w("이제", 0.0, 0.3)])
    assert len(events) == 1
    assert events[0].severity == "low"


def test_punctuation_normalized():
    events = detect_filler_events([_w("어,", 0.0, 0.2)])
    assert len(events) == 1


def test_normal_word_not_detected():
    events = detect_filler_events([_w("안녕하세요", 0.0, 0.5)])
    assert events == []


def test_repetition_detected():
    # "그 그" — 0.1s gap < 0.5s
    words = [_w("그", 0.0, 0.2), _w("그", 0.3, 0.5)]
    events = detect_filler_events(words)
    # 2 Tier 1 단일 + 1 반복 = 3
    repetitions = [e for e in events if " " in e.text]
    assert len(repetitions) == 1
    assert repetitions[0].severity == "mid"


def test_repetition_too_far_apart_not_merged():
    # 1.0s gap ≥ 0.5s → 반복 아님
    words = [_w("그", 0.0, 0.2), _w("그", 1.5, 1.7)]
    events = detect_filler_events(words)
    repetitions = [e for e in events if " " in e.text]
    assert len(repetitions) == 0


def test_empty_input():
    assert detect_filler_events([]) == []


def test_mixed_natural_with_fillers():
    words = [
        _w("안녕하세요", 0.0, 0.5),
        _w("음", 0.6, 0.7),
        _w("오늘은", 1.0, 1.3),
        _w("그", 1.4, 1.5),
        _w("타입스크립트를", 1.7, 2.3),
    ]
    events = detect_filler_events(words)
    # "음", "그" 두 개 검출
    detected_texts = [e.text for e in events]
    assert "음" in detected_texts
    assert "그" in detected_texts
    assert "안녕하세요" not in detected_texts
