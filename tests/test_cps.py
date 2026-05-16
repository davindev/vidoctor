"""CPS(발화 속도) 이상 검출 테스트 — net speech 윈도우 σ 분석."""

from tests._helpers import w
from vidoctor.audio.cps import detect_cps_anomalies
from vidoctor.graph.state import Word


def _normal_speech(duration_sec: float = 30.0, cps: float = 6.0) -> list[Word]:
    """평균 cps에 가까운 균질 발화 생성. 3글자 단어 + 짧은 휴지."""
    words = []
    chars_per_word = 3
    word_dur = chars_per_word / cps
    gap = 0.05
    t = 0.0
    while t + word_dur < duration_sec:
        words.append(w("테스트", t, t + word_dur))
        t += word_dur + gap
    return words


def _append_anomaly(
    words: list[Word], chars_per_word: int, target_cps: float, count: int, text: str
) -> None:
    word_dur = chars_per_word / target_cps
    t = words[-1].end + 0.1 if words else 0.0
    for _ in range(count):
        words.append(w(text, t, t + word_dur))
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
    # 50초 정상(6 CPS) + 느린 발화(1.5 CPS, 1글자). 의미 단어로 — filler는 cps 측정에서
    # 제외되므로 filler 어휘를 슬로우 fixture로 쓰면 검출 자체가 안 일어남.
    words = _normal_speech(duration_sec=50.0, cps=6.0)
    _append_anomaly(words, chars_per_word=1, target_cps=1.5, count=15, text="단")
    events = detect_cps_anomalies(words)
    slow = [e for e in events if e.kind == "too_slow"]
    assert len(slow) >= 1
    assert slow[0].cps < 3.0


def test_long_pause_excluded_from_cps():
    # 균일한 6 cps 발화 + 사이에 긴 휴지 → 휴지가 net speech에서 제외되면 윈도우 cps가
    # 일정해져 σ < MIN_STDEV로 컷오프. 휴지가 분모에 들어가면 휴지 걸친 윈도우 cps가
    # 떨어져 σ가 커지고 too_slow로 오탐. 즉 이 단언은 net speech 제외 동작의 간접 검증.
    words = []
    t = 0.0
    for _ in range(10):
        words.append(w("테스트", t, t + 0.5))
        t += 0.55
    t = 20.0
    for _ in range(20):
        words.append(w("테스트", t, t + 0.5))
        t += 0.55
    events = detect_cps_anomalies(words)
    too_slow = [e for e in events if e.kind == "too_slow"]
    assert too_slow == []


def test_filler_words_excluded_from_cps():
    # 정상 발화 + filler burst("음" 반복) → filler가 cps 분모/분자 모두에서 빠지면
    # filler 영역은 net_speech 부족으로 윈도우가 통계에 안 들어가고 too_slow 검출도 X.
    # filler가 cps에 포함되면 cps≈3 영역으로 잡혀 too_slow로 오분류.
    words = _normal_speech(duration_sec=50.0, cps=6.0)
    _append_anomaly(words, chars_per_word=1, target_cps=3.0, count=20, text="음")
    events = detect_cps_anomalies(words)
    slow_in_filler = [
        e for e in events if e.kind == "too_slow" and e.start >= 49.0
    ]
    assert slow_in_filler == []


def test_anomalies_localized_to_anomaly_segment():
    # 50초 정상 + 11초 빠름 → 검출된 fast 이벤트들은 모두 빠름 영역(50초+)과 겹쳐야 한다.
    # 5초 윈도우가 경계 직전부터 일부 빠름 영역을 흡수해 e.start는 50초 이전일 수 있지만,
    # e.end는 빠름 영역에 진입한 윈도우만 통계 임계를 넘으므로 50초를 넘어야 한다.
    words = _normal_speech(duration_sec=50.0, cps=6.0)
    _append_anomaly(words, chars_per_word=6, target_cps=14.0, count=25, text="테스트단어를")
    events = detect_cps_anomalies(words)
    fast = [e for e in events if e.kind == "too_fast"]
    assert len(fast) >= 1
    assert all(e.end > 50.0 for e in fast), "빠름 영역과 겹치지 않는 검출이 있음"
