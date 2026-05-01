"""시각 dead zone 검출.

화면 변화·발화 모두 정지된 구간을 검출. 단일 시각 신호(SSIM)로는 강의의 슬라이드
미세 변화·브이로그의 인물 미세 움직임에서 모두 취약 → ASR 무발화 구간과 결합한
시각·청각 2축 판정.

알고리즘:
1. 영상 프레임 2 fps로 샘플링 + 240p 다운스케일 + grayscale 변환
2. 인접 샘플 프레임 SSIM 계산 → SSIM ≥ 0.95 연속 구간을 시각 정적으로 판정
3. transcript 단어 사이 1초+ gap을 무발화 구간으로 추출
4. 두 구간의 교집합 → 카테고리별 최소 길이 임계 적용

상수는 휴리스틱 시작값. 골든셋 라벨링 후 우선 튜닝:
  1순위: CATEGORY_MIN_DURATION_SEC — 라벨링된 dead zone 길이 분포 기반
  2순위: SSIM_STATIC_THRESHOLD — 정적 vs 미세 변화의 분리 정확도 측정
  3순위: ASR_SILENCE_THRESHOLD_SEC — 한국어 발화 휴지 분포 기반
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from itertools import pairwise

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from vidoctor.graph.state import Category, DeadZoneEvent, Word

# 카테고리별 dead zone 최소 지속 시간. 강의는 슬라이드 한 장 오래 보여주는 게 정상이라
# 임계 보수적, 브이로그는 5초 정적도 사고에 가까워서 임계 좁음.
CATEGORY_MIN_DURATION_SEC: dict[Category, float] = {
    "lecture": 20.0,
    "vlog": 5.0,
    "other": 15.0,
}

# 프레임 샘플링 fps. 2 fps = 0.5초마다 1프레임.
# 더 높이면 정확도 ↑ but 처리 시간 ↑. dead zone은 5초+ 단위라 2 fps로 충분.
FRAME_SAMPLE_FPS = 2.0

# SSIM ≥ 이 값이면 시각 정적으로 판정. mp4v 압축 노이즈 감안한 보수적 값.
SSIM_STATIC_THRESHOLD = 0.95

# 단어 사이 gap > 이 값이면 무발화 구간. 자연스러운 휴지(0.5s 이내)는 발화의 일부로 간주.
ASR_SILENCE_THRESHOLD_SEC = 1.0

# SSIM 계산용 다운스케일 해상도. dead zone 검출에 고해상도 불필요.
DOWNSAMPLE_HEIGHT = 240


@dataclass(frozen=True)
class _Interval:
    start: float
    end: float


def _silent_intervals(words: list[Word], video_duration: float) -> list[_Interval]:
    """transcript에서 무발화 구간 추출.

    영상 시작 ~ 첫 단어 / 단어 사이 긴 gap / 마지막 단어 ~ 영상 끝.
    transcript가 비어 있으면 영상 전체가 무발화.
    """
    if not words:
        return [_Interval(0.0, video_duration)] if video_duration > 0 else []

    intervals: list[_Interval] = []

    if words[0].start > ASR_SILENCE_THRESHOLD_SEC:
        intervals.append(_Interval(0.0, words[0].start))

    for prev, curr in pairwise(words):
        gap = curr.start - prev.end
        if gap > ASR_SILENCE_THRESHOLD_SEC:
            intervals.append(_Interval(prev.end, curr.start))

    if video_duration - words[-1].end > ASR_SILENCE_THRESHOLD_SEC:
        intervals.append(_Interval(words[-1].end, video_duration))

    return intervals


def _static_intervals(video_path: str) -> tuple[list[_Interval], float]:
    """영상에서 SSIM 기반 시각 정적 구간 추출. (intervals, total_duration) 반환."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"미디어 파일 열기 실패: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0

        sample_step = max(int(round(fps / FRAME_SAMPLE_FPS)), 1)

        intervals: list[_Interval] = []
        prev_gray: np.ndarray | None = None
        prev_time = 0.0
        is_static = False
        static_start = 0.0
        static_end = 0.0

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_step != 0:
                frame_idx += 1
                continue

            curr_time = frame_idx / fps if fps > 0 else 0.0

            h, w = frame.shape[:2]
            if h > DOWNSAMPLE_HEIGHT:
                scale = DOWNSAMPLE_HEIGHT / h
                frame = cv2.resize(frame, (int(w * scale), DOWNSAMPLE_HEIGHT))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                # full=False가 기본이라 float 반환이지만 pyright stub은 union으로 추론
                ssim: float = structural_similarity(prev_gray, gray, data_range=255)  # pyright: ignore[reportAssignmentType]
                if ssim >= SSIM_STATIC_THRESHOLD:
                    # SSIM(prev, curr) ≥ 임계면 [prev_time, curr_time]이 정적.
                    # 정적 구간 끝점은 마지막으로 임계 통과한 curr_time.
                    if not is_static:
                        static_start = prev_time
                        is_static = True
                    static_end = curr_time
                elif is_static:
                    intervals.append(_Interval(static_start, static_end))
                    is_static = False

            prev_gray = gray
            prev_time = curr_time
            frame_idx += 1

        if is_static:
            intervals.append(_Interval(static_start, static_end))

        return intervals, duration
    finally:
        cap.release()


def _intersect_intervals(a: list[_Interval], b: list[_Interval]) -> list[_Interval]:
    """정렬된 두 interval 리스트의 교집합. 투포인터 O(n+m)."""
    result: list[_Interval] = []
    i = j = 0
    while i < len(a) and j < len(b):
        ai, bj = a[i], b[j]
        start = max(ai.start, bj.start)
        end = min(ai.end, bj.end)
        if start < end:
            result.append(_Interval(start, end))
        if ai.end < bj.end:
            i += 1
        else:
            j += 1
    return result


def _detect_dead_zone_sync(
    video_path: str,
    transcript: list[Word],
    category: Category,
) -> list[DeadZoneEvent]:
    static, duration = _static_intervals(video_path)
    silent = _silent_intervals(transcript, duration)
    overlap = _intersect_intervals(static, silent)

    min_duration = CATEGORY_MIN_DURATION_SEC[category]

    events: list[DeadZoneEvent] = []
    for iv in overlap:
        if iv.end - iv.start >= min_duration:
            events.append(DeadZoneEvent(start=iv.start, end=iv.end, severity="mid"))
    return events


async def detect_dead_zone_events(
    video_path: str,
    transcript: list[Word],
    category: Category,
) -> list[DeadZoneEvent]:
    """영상 + transcript + 카테고리 → dead zone 이벤트 리스트.

    OpenCV·SSIM 처리는 sync·CPU bound이라 to_thread로 분리해 이벤트 루프 차단 방지.
    """
    return await asyncio.to_thread(_detect_dead_zone_sync, video_path, transcript, category)
