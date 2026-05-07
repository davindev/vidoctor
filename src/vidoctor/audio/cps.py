"""Net CPS(Characters Per Second) 슬라이딩 윈도우 이상 검출.

5초 윈도우 / 1초 스텝. 단어 사이 200ms+ pause는 제외한 순수 발화 시간으로 글자수를 나눔.
한국어는 영어 WPM이 부적합 → 음절 기반 CPS가 자연스러움.

판정: 영상 평균 ±SIGMA_THRESHOLD σ 이탈을 후보로 삼는다. cps 단독으론 정상 발화의 단순
빠름과 라벨러가 인지하는 "속사포"가 cps 분포에서 부분 겹쳐 분리 한계가 있음을 골든셋
sweep으로 확인. F0(피치) 신호가 있으면 too_fast 판정에 AND 조건으로 결합 — 라벨러 인지의
"속사포"는 톤 상승 동반 패턴이 일관돼 cps + F0 multi-feature가 P를 끌어올린다.

filler 단어는 cps 측정에서 제외. "음·어" 등 disfluency는 의미 발화가 아니라 cps 비율을
왜곡(짧은 글자수 + 짧은 시간 → 인공적으로 낮은 cps)해 too_slow false positive를 만든다.

평탄 영상 가드: σ가 MIN_STDEV 미만이면(균질 발화) 임계가 좁아 노이즈도 잡혀 검출을
스킵.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from vidoctor.audio.filler import FILLERS, normalize_word
from vidoctor.audio.pitch import WindowPitch
from vidoctor.graph.state import CPSEvent, Word

# 윈도우 길이 5초. 25~35자 / 6~8 단어 표본 → 통계 안정 + 사용자 인지 단위 균형.
# 음성처리 연구에서 speech rate 측정에 3~10초가 일반.
WINDOW_SEC = 5.0

# 1초 스텝 → 5초 윈도우와 80% 겹침. 인접 병합으로 검출 해상도 안정화.
STEP_SEC = 1.0

# 자연 단어간 휴지와 의미 있는 멈춤의 경계.
#   - 자연 단어간 휴지: 50~200ms (호흡·자음 클로저·연접 음운 처리)
#   - 강조성 멈춤: 200~500ms (운율·강조)
#   - editing region(머뭇거림성 반복·수정 사이): 평균 300~700ms — Shriberg(1994)
# 200ms는 자연 휴지의 상한이자 의미 있는 멈춤의 시작점. 이 미만은 발화의 일부로
# 분모(net speech)에 포함, 이상은 "진짜 멈춤"으로 분모에서 제외해 cps를 순수 발화 속도로
# 측정. filler.py의 REPETITION_GAP_THRESHOLD_SEC(0.5)도 같은 분포에서 editing region
# 중간점으로 채택된 값이라 두 임계는 동일 근거 체계.
PAUSE_THRESHOLD_SEC = 0.2

# cps 분포의 σ 임계. 양방향 동일 — vlog 골든셋 sweep(σ 0.6~2.0)에서 σ=1.5가 P/R 균형
# sweet spot으로 확인. σ를 좁히면 R↑ P↓, 넓히면 P 동일 R↓로 F1 모두 하락.
SIGMA_THRESHOLD = 1.5

# 평탄 영상 컷오프(cps 단위). σ가 작은 영상은 z 임계가 좁아 수치 노이즈도 잡히므로 스킵.
MIN_STDEV = 0.5

# F0 결합 모드에서 too_fast 후보가 cps 임계를 통과한 뒤 추가로 만족해야 하는 F0 z 임계.
# vlog sweep에서 0.8이 P 0.6 / F1 0.46 sweet spot. 더 strict하면 R 하락, 더 관대하면 FP↑.
F0_AND_SIGMA = 0.8

# 5초 윈도우 중 0.5초 미만 발화 = 90%+ 침묵. 단어 1~2개로 cps 계산 noise → 윈도우 자체 skip.
MIN_NET_SPEECH_SEC = 0.5

# stdev 신뢰성 최소 표본. 3개+부터 분포 추정 의미. 영상 길이 환산: 약 7초+ 발화 필요.
MIN_WINDOWS_FOR_STATS = 3


@dataclass(frozen=True)
class _Window:
    start: float
    end: float
    cps: float


def _is_filler(word: Word) -> bool:
    """filler 사전과 단어 normalize 결과가 일치하면 filler로 간주."""
    return normalize_word(word.text) in FILLERS


def _net_speech_seconds(words: list[Word], start: float, end: float) -> float:
    """윈도우 내 의미 발화 시간 + 200ms 미만 짧은 휴지의 합.

    긴 휴지(≥200ms)는 "진짜 멈춤"으로 보고 제외. filler 단어는 의미 발화가 아니라
    net speech에서도 제외한다 — 그 사이 시간(filler 단어 길이 + 인접 휴지)은 자동으로
    긴 휴지로 분류되어 분모에서 빠진다.
    """
    in_window = sorted(
        (
            w for w in words
            if w.end > start and w.start < end and not _is_filler(w)
        ),
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
    """윈도우와 겹치는 의미 단어 글자수 (윈도우 경계에 걸친 단어는 비례 배분).

    filler 단어는 _net_speech_seconds와 짝을 맞춰 분자에서도 제외해 cps 비율 일관 유지.
    """
    total = 0.0
    for w in words:
        if w.end <= start or w.start >= end:
            continue
        if _is_filler(w):
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


@dataclass(frozen=True)
class _F0Baseline:
    """영상 단위 F0 통계 baseline. mean·range 각각 (mean, std) 튜플."""

    mean: tuple[float, float]
    range: tuple[float, float]


def _judge(
    window: _Window,
    cps_mean: float,
    cps_std: float,
    pitch: WindowPitch | None,
    f0_baseline: _F0Baseline | None,
) -> CPSEvent | None:
    """cps z-score 기준 판정. F0 baseline이 주어지면 too_fast에 한해 AND 조건 추가.

    too_slow는 F0 신호가 라벨러 인지와 약하다는 분석 결과에 따라 cps 단독 판정 유지.
    """
    cps_z = (window.cps - cps_mean) / cps_std
    if cps_z > SIGMA_THRESHOLD:
        if pitch is None or f0_baseline is None:
            kind: Literal["too_fast", "too_slow"] = "too_fast"
        else:
            f0m_z = (pitch.f0_mean - f0_baseline.mean[0]) / f0_baseline.mean[1]
            f0r_z = (pitch.f0_range - f0_baseline.range[0]) / f0_baseline.range[1]
            if f0m_z <= F0_AND_SIGMA and f0r_z <= F0_AND_SIGMA:
                return None
            kind = "too_fast"
    elif cps_z < -SIGMA_THRESHOLD:
        kind = "too_slow"
    else:
        return None
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


def detect_cps_anomalies(
    words: list[Word],
    pitch_features: list[WindowPitch | None] | None = None,
    windows: list[_Window] | None = None,
) -> list[CPSEvent]:
    """슬라이딩 윈도우로 CPS 이상 구간 검출.

    pitch_features가 주어지면 too_fast 판정에 cps z AND F0 z 결합 적용. None이면 cps z
    단독 판정. windows를 호출자가 미리 계산해 넘기면 재계산 회피 — pitch_features와 동일
    윈도우 정의를 공유해야 하는 호출 경로(`detect_cps_with_audio`)에서 사용.
    """
    if windows is None:
        windows = _sliding_windows(words)
    if len(windows) < MIN_WINDOWS_FOR_STATS:
        return []
    if pitch_features is not None and len(pitch_features) != len(windows):
        raise ValueError(
            f"pitch_features 길이({len(pitch_features)})가 윈도우({len(windows)})와 불일치"
        )

    cps_values = [w.cps for w in windows]
    cps_mean = statistics.mean(cps_values)
    cps_std = statistics.stdev(cps_values)
    if cps_std < MIN_STDEV:
        return []

    f0_baseline: _F0Baseline | None = None
    if pitch_features is not None:
        f0_means = [p.f0_mean for p in pitch_features if p is not None]
        f0_ranges = [p.f0_range for p in pitch_features if p is not None]
        if len(f0_means) >= MIN_WINDOWS_FOR_STATS:
            f0_baseline = _F0Baseline(
                mean=(statistics.mean(f0_means), statistics.stdev(f0_means)),
                range=(statistics.mean(f0_ranges), statistics.stdev(f0_ranges)),
            )

    raw = []
    for i, w in enumerate(windows):
        pitch = pitch_features[i] if pitch_features is not None else None
        ev = _judge(w, cps_mean, cps_std, pitch, f0_baseline)
        if ev is not None:
            raw.append(ev)
    return _merge_adjacent(raw)


def detect_cps_with_audio(words: list[Word], audio_path: str) -> list[CPSEvent]:
    """오디오 path 받아 F0 추출 + multi-feature detector 일괄 처리.

    호출자가 윈도우 정의·F0 추출·detector 호출 정합을 직접 챙기지 않게 캡슐화 —
    `_sliding_windows`를 cross-module로 노출하지 않고 한 함수가 책임진다.
    """
    from vidoctor.audio.pitch import extract_pitch_track, window_pitch_features

    windows = _sliding_windows(words)
    if not windows:
        return []
    f0, times = extract_pitch_track(audio_path)
    pitch_features = window_pitch_features(
        f0, times, [(w.start, w.end) for w in windows]
    )
    return detect_cps_anomalies(words, pitch_features=pitch_features, windows=windows)
