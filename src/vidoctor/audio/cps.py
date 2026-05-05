"""Net CPS(Characters Per Second) 슬라이딩 윈도우 이상 검출.

5초 윈도우 / 1초 스텝. 단어 사이 200ms+ pause는 제외한 순수 발화 시간으로 글자수를 나눔.
한국어는 영어 WPM이 부적합 → 음절 기반 CPS가 자연스러움.

판정 철학: "이 영상에서 튀는 구간"만 이상으로 본다. 절대 임계(<3, >9)는 화자별·영상
종류별 정상 속도 편차를 무시하는 자의적 보편값이라 제거. 영상 평균 대비 ±1.5σ 이탈만 사용.
kind는 평균 대비 방향으로 결정(cps > mean → too_fast).
인접한 동종 이상 윈도우는 한 구간으로 병합.

평탄 영상 가드: σ가 매우 작은 영상(균질 발화)에선 ±1.5σ 임계가 좁게 형성돼 단어 길이
비례 배분 등에서 발생하는 수치 노이즈도 이상으로 잡힘. MIN_STDEV 미만이면 검출 자체를
스킵해 false positive 방지.

상수는 모두 휴리스틱 시작값. 골든셋 라벨링 후 우선 튜닝:
  1순위: SIGMA_THRESHOLD — ±1.2σ/±1.5σ/±2σ로 precision/recall 트레이드 탐색.
         현재 1.5가 sweet spot(±1.2σ는 vlog FP 폭증).
  2순위: MIN_STDEV — 실데이터(lecture σ=2.19, vlog σ=2.45)엔 영향 없으나 균질 발화
         방어용. 평탄에 가까운 영상 추가 시 재검토.
  3순위: WINDOW_SEC — 3s vs 7s 비교해서 F1 최대화
  4순위: PAUSE_THRESHOLD_SEC, MIN_NET_SPEECH_SEC — 한국어 발화 패턴 적응
  나머지(MIN_WINDOWS_FOR_STATS, STEP_SEC)는 통계 표준값이라 유지.
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

# ±1.5σ ≈ 정상 범위 87%. ±2σ(95%)는 vlog 라벨 대비 recall이 0.125로 너무 보수적이라
# ±1.5σ로 완화. ±1.2σ도 시도했으나 vlog FP가 7→15로 폭증해 F1·macro_f1 모두 하락 →
# ±1.5σ가 sweet spot으로 확인. ±1σ(68%)는 자연 변동까지 잡을 위험. 골든셋 확장 후 재튜닝.
SIGMA_THRESHOLD = 1.5

# 평탄 영상 컷오프(cps 단위). σ가 작은 영상은 ±1.5σ 임계가 좁아 수치 노이즈도 잡히므로
# 검출 자체를 스킵. 실데이터(lecture 2.19, vlog 2.45)엔 영향 없지만 균질 발화 합성 케이스
# (test_normal_speech_no_anomaly)에서 false positive 차단을 검증. 갱신은 평탄 영상 라벨 후.
MIN_STDEV = 0.5

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
    if abs(window.cps - mean) <= SIGMA_THRESHOLD * std:
        return None
    kind: Literal["too_fast", "too_slow"] = "too_fast" if window.cps > mean else "too_slow"
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

    윈도우 수가 부족하거나(통계 못 냄), σ가 MIN_STDEV 미만이면(평탄 영상) 빈 리스트 반환.
    """
    windows = _sliding_windows(words)
    if len(windows) < MIN_WINDOWS_FOR_STATS:
        return []

    cps_values = [w.cps for w in windows]
    mean = statistics.mean(cps_values)
    std = statistics.stdev(cps_values)
    if std < MIN_STDEV:
        return []

    raw = [ev for ev in (_judge(w, mean, std) for w in windows) if ev is not None]
    return _merge_adjacent(raw)
