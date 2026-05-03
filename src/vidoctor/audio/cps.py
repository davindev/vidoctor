"""Net CPS(Characters Per Second) 슬라이딩 윈도우 이상 검출.

5초 윈도우 / 1초 스텝. 단어 사이 200ms+ pause는 제외한 순수 발화 시간으로 글자수를 나눔.
한국어는 영어 WPM이 부적합 → 음절 기반 CPS가 자연스러움.

판정: 절대 기준(<3 or >9 CPS) AND 영상 평균 ±2σ 이탈 동시 충족만 이상으로 표시.
인접한 동종 이상 윈도우는 한 구간으로 병합.

상수는 모두 휴리스틱 시작값. 골든셋 라벨링 후 우선 튜닝:
  1순위: ABSOLUTE_FAST_CPS / ABSOLUTE_SLOW_CPS — 라벨링된 빠름·느림 구간의 실측 분포 기반
  2순위: WINDOW_SEC — 3s vs 7s 비교해서 F1 최대화
  3순위: PAUSE_THRESHOLD_SEC, MIN_NET_SPEECH_SEC — 한국어 발화 패턴 적응
  나머지(SIGMA_THRESHOLD, MIN_WINDOWS_FOR_STATS, STEP_SEC)는 통계 표준값이라 유지.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from vidoctor.graph.state import CPSEvent, Word

# 윈도우 길이 5초. 25~35자 / 6~8 단어 표본 → 통계 안정 + 사용자 인지 단위 균형.
# 음성처리 연구에서 speech rate 측정에 3~10초가 일반.
WINDOW_SEC = 5.0

# 1초 스텝 → 5초 윈도우와 80% 겹침. 인접 병합으로 검출 해상도 안정화.
STEP_SEC = 1.0

# 자연 단어간 휴지(50~150ms)와 강조성 멈춤(200~500ms)의 경계.
# Shriberg(1994) 등 음성처리 연구의 통용 임계. 이 미만은 발화의 일부, 이상은 진짜 멈춤.
PAUSE_THRESHOLD_SEC = 0.2

# 한국어 평균 5~7 CPS, 7~9 빠른 편, 9+ 청자 따라가기 부담 시작.
ABSOLUTE_FAST_CPS = 9.0

# 3 CPS 미만은 발화 흐름 끊김 인지 경계 (한국어는 빠른 쪽보다 느린 쪽 임계가 좁음).
ABSOLUTE_SLOW_CPS = 3.0

# ±2σ = 정규분포 95% 정상 범위. 통계 outlier 검출의 사실상 표준값.
# ±1σ는 false positive 너무 많고 ±3σ는 너무 보수적.
SIGMA_THRESHOLD = 2.0

# 5초 윈도우 중 0.5초 미만 발화 = 90%+ 침묵. 단어 1~2개로 cps 계산 noise → 윈도우 자체 skip.
MIN_NET_SPEECH_SEC = 0.5

# stdev 신뢰성 최소 표본. 2개는 차이의 절반에 불과. 3개+부터 분포 추정 의미.
# 영상 길이 환산: 약 7초+ 발화 필요.
MIN_WINDOWS_FOR_STATS = 3


@dataclass(frozen=True)
class _Window:
    start: float
    end: float
    cps: float


def _net_speech_seconds(words: list[Word], start: float, end: float) -> float:
    """윈도우 내 단어 발화 시간 + 200ms 미만 짧은 휴지의 합.

    긴 휴지(≥200ms)는 "진짜 멈춤"으로 보고 제외 → 강조·자연스러운 톤 변화로 인한
    false positive 감소.
    """
    in_window = sorted(
        (w for w in words if w.end > start and w.start < end),
        key=lambda x: x.start,
    )
    if not in_window:
        return 0.0

    speech = 0.0
    for w in in_window:
        s = max(w.start, start)
        e = min(w.end, end)
        if e > s:
            speech += e - s

    short_pauses = 0.0
    for prev, curr in pairwise(in_window):
        gap = curr.start - prev.end
        if 0.0 < gap < PAUSE_THRESHOLD_SEC:
            short_pauses += gap

    return speech + short_pauses


def _char_count(words: list[Word], start: float, end: float) -> float:
    """윈도우와 겹치는 단어 글자수 (윈도우 경계에 걸친 단어는 비례 배분)."""
    total = 0.0
    for w in words:
        if w.end <= start or w.start >= end:
            continue
        word_dur = max(w.end - w.start, 1e-6)
        overlap = min(w.end, end) - max(w.start, start)
        total += len(w.text) * (overlap / word_dur)
    return total


def _sliding_windows(words: list[Word]) -> list[_Window]:
    if not words:
        return []

    audio_start = words[0].start
    audio_end = words[-1].end

    windows: list[_Window] = []
    t = audio_start
    while t < audio_end:
        win_end = t + WINDOW_SEC
        net_time = _net_speech_seconds(words, t, win_end)
        if net_time >= MIN_NET_SPEECH_SEC:
            chars = _char_count(words, t, win_end)
            if chars > 0:
                windows.append(_Window(start=t, end=win_end, cps=chars / net_time))
        t += STEP_SEC

    return windows


def _judge(window: _Window, mean: float, std: float) -> CPSEvent | None:
    abs_fast = window.cps > ABSOLUTE_FAST_CPS
    abs_slow = window.cps < ABSOLUTE_SLOW_CPS
    if not (abs_fast or abs_slow):
        return None

    rel_anomaly = std > 0 and abs(window.cps - mean) > SIGMA_THRESHOLD * std
    if not rel_anomaly:
        return None

    kind: Literal["too_fast", "too_slow"] = "too_fast" if abs_fast else "too_slow"
    return CPSEvent(start=window.start, end=window.end, cps=window.cps, kind=kind)


def _merge_adjacent(events: list[CPSEvent]) -> list[CPSEvent]:
    """동종(too_fast/too_slow) 인접 이벤트를 한 구간으로 병합. cps는 길이 가중 평균."""
    if not events:
        return []

    merged = [events[0]]
    for ev in events[1:]:
        last = merged[-1]
        if last.kind == ev.kind and ev.start - last.end < STEP_SEC + 0.1:
            last_dur = last.end - last.start
            ev_dur = ev.end - ev.start
            weighted_cps = (last.cps * last_dur + ev.cps * ev_dur) / (last_dur + ev_dur)
            merged[-1] = CPSEvent(
                start=last.start,
                end=ev.end,
                cps=weighted_cps,
                kind=last.kind,
            )
        else:
            merged.append(ev)
    return merged


def detect_cps_anomalies(words: list[Word]) -> list[CPSEvent]:
    """슬라이딩 윈도우로 CPS 이상 구간 검출.

    윈도우 수가 적거나 std=0이면 통계 기반 판정 불가 → 빈 리스트 반환.
    """
    windows = _sliding_windows(words)
    if len(windows) < MIN_WINDOWS_FOR_STATS:
        return []

    cps_values = [w.cps for w in windows]
    mean = statistics.mean(cps_values)
    std = statistics.stdev(cps_values)

    raw = [ev for ev in (_judge(w, mean, std) for w in windows) if ev is not None]
    return _merge_adjacent(raw)
