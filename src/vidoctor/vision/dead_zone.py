"""시각 dead zone 검출.

화면 변화·발화 모두 정지된 구간 검출.

1. **Silero VAD**로 무발화 구간 추출. 환경 소음(차·바람·음악)은 비음성으로 자동 무시.
   배경의 다른 사람 목소리는 발화로 잡힘 — 화자 분리는 별도 차원.
2. 무발화 구간 중 카테고리별 `MIN_DURATION_SEC` 이상을 dead zone 후보.
3. **Optical flow magnitude의 per-frame max** 시계열로 후보 안 화면 움직임 측정. 평균은
   화면 작은 영역(예: lecture 우하단 페이스캠) 움직임이 큰 정적 영역(슬라이드)에 묻혀
   사용자 인지 "화면 움직임"을 못 잡음. per-frame max는 한 픽셀이라도 크게 움직이면
   잡혀 작은 영역 움직임에 robust.
4. 후보 안 per-frame max 시계열의 median이 카테고리별 임계 이하일 때 시각 정적으로 인정.

카테고리별 임계가 갈리는 이유: lecture는 삼각대 고정이라 정적 floor가 0에 가깝고,
vlog는 핸드헬드라 정적이어도 카메라 미세 흔들림으로 floor가 2~3 깔림. 단일 절대 임계로
양쪽 baseline 위 신호 분리 불가능.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
import torch
import whisperx
from silero_vad import get_speech_timestamps, load_silero_vad

from vidoctor.graph.state import Category, DeadZoneEvent

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CategoryConfig:
    min_duration_sec: float
    flow_max_threshold: float


# 라벨 표본 분리 sweet spot:
#   lecture: TP 0.32 / 페이스캠 FP 0.98 → 임계 0.5
#   vlog:    TP 2.3·3.4 / FP_가짜 21.6 → 임계 5.0
#   other:   라벨 없음 — vlog 기준 보수적
CATEGORY_CONFIG: dict[Category, _CategoryConfig] = {
    "lecture": _CategoryConfig(min_duration_sec=5.0, flow_max_threshold=0.5),
    "vlog": _CategoryConfig(min_duration_sec=5.0, flow_max_threshold=5.0),
    "other": _CategoryConfig(min_duration_sec=5.0, flow_max_threshold=5.0),
}

FRAME_SAMPLE_FPS = 2.0
DOWNSAMPLE_HEIGHT = 240

VAD_SAMPLE_RATE = 16000
VAD_MIN_SILENCE_MS = 1000

# Farneback dense flow 파라미터 (OpenCV 권장 default 그대로). cv2가 kwargs 미지원이라
# named 상수로 분리해 호출부 가독성 확보.
_FARNEBACK_PYR_SCALE = 0.5
_FARNEBACK_LEVELS = 3
_FARNEBACK_WINSIZE = 15
_FARNEBACK_ITERATIONS = 3
_FARNEBACK_POLY_N = 5
_FARNEBACK_POLY_SIGMA = 1.2
_FARNEBACK_FLAGS = 0

# whisperx.load_audio 내부 ffmpeg가 audio 트랙 없는 영상에 대해 던지는 메시지 marker.
# 이 marker가 포함된 RuntimeError만 무성 영상으로 fallback하고, 다른 RuntimeError(파일
# 손상·ffmpeg 미설치 등)는 그대로 raise해 진단 가능하게 둔다.
_FFMPEG_NO_STREAM_MARKER = "Output file does not contain any stream"


@dataclass(frozen=True)
class _Interval:
    start: float
    end: float


@lru_cache(maxsize=1)
def _vad_model() -> Any:
    return load_silero_vad()


def _silent_intervals_from_audio(
    audio: np.ndarray, video_duration: float
) -> list[_Interval]:
    """Silero VAD로 발화 구간 추출 → 영상 - 발화 = 무발화 구간."""
    if audio.size == 0:
        return [_Interval(0.0, video_duration)] if video_duration > 0 else []

    audio_tensor = torch.from_numpy(audio).float()
    speech_ts: list[dict[str, float]] = get_speech_timestamps(
        audio_tensor,
        _vad_model(),
        sampling_rate=VAD_SAMPLE_RATE,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        return_seconds=True,
    )

    if not speech_ts:
        return [_Interval(0.0, video_duration)] if video_duration > 0 else []

    silents: list[_Interval] = []
    prev_end = 0.0
    for t in speech_ts:
        s, e = float(t["start"]), float(t["end"])
        if s > prev_end:
            silents.append(_Interval(prev_end, s))
        prev_end = max(prev_end, e)
    if video_duration > prev_end:
        silents.append(_Interval(prev_end, video_duration))
    return silents


def _flow_series(video_path: str) -> tuple[np.ndarray, np.ndarray, float]:
    """샘플 프레임 인접쌍 optical flow magnitude per-frame max 시계열.

    Farneback dense flow → 픽셀별 (dx, dy) 벡터 → magnitude → 프레임 안 max.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"미디어 파일 열기 실패: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0
        sample_step = max(int(round(fps / FRAME_SAMPLE_FPS)), 1)

        curr_times: list[float] = []
        flows: list[float] = []
        prev_gray: np.ndarray | None = None

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
                # cv2 stub은 flow=None을 거부 — 빈 ndarray placeholder 전달, OpenCV가 새 buffer alloc.
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    gray,
                    np.empty(0, dtype=np.float32),
                    _FARNEBACK_PYR_SCALE,
                    _FARNEBACK_LEVELS,
                    _FARNEBACK_WINSIZE,
                    _FARNEBACK_ITERATIONS,
                    _FARNEBACK_POLY_N,
                    _FARNEBACK_POLY_SIGMA,
                    _FARNEBACK_FLAGS,
                )
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                curr_times.append(curr_time)
                flows.append(float(mag.max()))

            prev_gray = gray
            frame_idx += 1
    finally:
        cap.release()

    return (
        np.asarray(curr_times, dtype=np.float64),
        np.asarray(flows, dtype=np.float64),
        duration,
    )


def _flow_median_in(
    curr_times: np.ndarray,
    flows: np.ndarray,
    start: float,
    end: float,
) -> float | None:
    """주어진 시간 구간 안 flow 시계열의 median. 샘플 없으면 None."""
    mask = (curr_times >= start) & (curr_times <= end)
    if not mask.any():
        return None
    return float(np.median(flows[mask]))


def _load_audio_or_empty(video_path: str) -> np.ndarray:
    """audio 트랙 없는 영상은 빈 array fallback. 다른 ffmpeg 에러는 그대로 raise."""
    try:
        return whisperx.load_audio(video_path)
    except RuntimeError as e:
        if _FFMPEG_NO_STREAM_MARKER in str(e):
            _log.warning("audio track missing; dead_zone VAD step skipped: video=%s", video_path)
            return np.array([], dtype=np.float32)
        raise


def _detect_dead_zone_sync(
    video_path: str,
    category: Category,
) -> list[DeadZoneEvent]:
    curr_times, flows, duration = _flow_series(video_path)
    audio = _load_audio_or_empty(video_path)
    silent = _silent_intervals_from_audio(audio, duration)
    cfg = CATEGORY_CONFIG[category]

    events: list[DeadZoneEvent] = []
    for iv in silent:
        if iv.end - iv.start < cfg.min_duration_sec:
            continue
        median = _flow_median_in(curr_times, flows, iv.start, iv.end)
        if median is None or median > cfg.flow_max_threshold:
            continue
        events.append(DeadZoneEvent(start=iv.start, end=iv.end))
    return events


async def detect_dead_zone_events(
    video_path: str,
    category: Category,
) -> list[DeadZoneEvent]:
    """영상 + 카테고리 → dead zone 이벤트 리스트.

    OpenCV·flow·VAD 처리는 sync·CPU bound이라 to_thread로 분리해 이벤트 루프 차단 방지.
    """
    return await asyncio.to_thread(_detect_dead_zone_sync, video_path, category)
