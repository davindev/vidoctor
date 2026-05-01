from vidoctor.audio.cps import detect_cps_anomalies
from vidoctor.graph.state import Word


def _w(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end, score=0.9)


def _normal_speech(duration_sec: float = 30.0, cps: float = 6.0) -> list[Word]:
    """평균 cps에 가까운 균질 발화 생성. 3글자 단어 + 짧은 휴지."""
    words = []
    chars_per_word = 3
    word_dur = chars_per_word / cps
    gap = 0.05
    t = 0.0
    while t + word_dur < duration_sec:
        words.append(_w("테스트", t, t + word_dur))
        t += word_dur + gap
    return words


def _append_anomaly(
    words: list[Word], chars_per_word: int, target_cps: float, count: int, text: str
) -> None:
    word_dur = chars_per_word / target_cps
    t = words[-1].end + 0.1 if words else 0.0
    for _ in range(count):
        words.append(_w(text, t, t + word_dur))
        t += word_dur + 0.02


def test_empty_input():
    assert detect_cps_anomalies([]) == []


def test_short_audio_no_windows():
    # 5초 미만 → 통계 윈도우 부족
    assert detect_cps_anomalies(_normal_speech(duration_sec=4.0)) == []


def test_normal_speech_no_anomaly():
    words = _normal_speech(duration_sec=30.0, cps=6.0)
    events = detect_cps_anomalies(words)
    assert events == []


def test_too_fast_segment_detected():
    # 50초 정상(6 CPS) + 11초 빠른 발화(14 CPS, 6글자 단어)
    words = _normal_speech(duration_sec=50.0, cps=6.0)
    _append_anomaly(words, chars_per_word=6, target_cps=14.0, count=25, text="테스트단어를")
    events = detect_cps_anomalies(words)
    fast = [e for e in events if e.kind == "too_fast"]
    assert len(fast) >= 1
    assert fast[0].cps > 9.0


def test_too_slow_segment_detected():
    # 50초 정상(6 CPS) + 10초 느린 발화(1.5 CPS, 1글자)
    words = _normal_speech(duration_sec=50.0, cps=6.0)
    _append_anomaly(words, chars_per_word=1, target_cps=1.5, count=15, text="음")
    events = detect_cps_anomalies(words)
    slow = [e for e in events if e.kind == "too_slow"]
    assert len(slow) >= 1
    assert slow[0].cps < 3.0


def test_long_pause_excluded_from_cps():
    # 짧은 발화 후 10초 휴지 → 휴지 구간은 net speech 부족으로 윈도우 자체가 거의 안 만들어짐.
    # too_slow 오탐 없어야 (휴지가 cps 계산에서 제외되므로)
    words = [_w("안녕하세요여러분", 0.0, 2.0)]
    t = 12.0
    for _ in range(20):
        words.append(_w("테스트", t, t + 0.5))
        t += 0.55
    events = detect_cps_anomalies(words)
    too_slow = [e for e in events if e.kind == "too_slow"]
    assert too_slow == []


def test_anomalies_localized_to_anomaly_segment():
    # 50초 정상 + 11초 빠름 → 검출된 fast 이벤트들은 모두 50초 이후에 발생해야
    # (병합 정도는 통계 분포에 따라 달라짐 — 핵심은 "이상 구간이 검출되고 위치가 맞다")
    words = _normal_speech(duration_sec=50.0, cps=6.0)
    _append_anomaly(words, chars_per_word=6, target_cps=14.0, count=25, text="테스트단어를")
    events = detect_cps_anomalies(words)
    fast = [e for e in events if e.kind == "too_fast"]
    assert len(fast) >= 1
    assert all(e.start >= 49.0 for e in fast), "이상 구간이 정상 구간으로 새어 들어감"
