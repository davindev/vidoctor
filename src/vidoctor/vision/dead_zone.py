"""시각 dead zone 검출 — Silero VAD 무발화 + Optical flow per-frame max gate.

Silero VAD 무발화 구간 중 카테고리별 min_duration 이상 + flow median이 임계 이하인
구간만 채택. per-frame max 사용 근거: 평균은 큰 정적 영역(슬라이드)에 작은 영역(페이스캠)
움직임이 묻혀 사용자 인지 "화면 움직임"을 못 잡음. 카테고리별 임계는 lecture 삼각대 floor
0 근처 vs vlog 핸드헬드 baseline 2~3 — 단일 절대 임계로는 분리 불가능.
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
from vidoctor.vision._capture import open_capture

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
class SilentInterval:
    """VAD가 추출한 무발화 시간 구간 (초)."""

    start: float
    end: float


@lru_cache(maxsize=1)
def _vad_model() -> Any:
    return load_silero_vad()


def silent_intervals_from_audio(
    audio: np.ndarray, video_duration: float
) -> list[SilentInterval]:
    """Silero VAD로 발화 구간 추출 → 영상 - 발화 = 무발화 구간."""
    if audio.size == 0:
        return [SilentInterval(0.0, video_duration)] if video_duration > 0 else []

    audio_tensor = torch.from_numpy(audio).float()  # pyright: ignore[reportPrivateImportUsage]
    speech_ts: list[dict[str, float]] = get_speech_timestamps(
        audio_tensor,
        _vad_model(),
        sampling_rate=VAD_SAMPLE_RATE,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        return_seconds=True,
    )

    if not speech_ts:
        return [SilentInterval(0.0, video_duration)] if video_duration > 0 else []

    silents: list[SilentInterval] = []
    prev_end = 0.0
    for t in speech_ts:
        s, e = float(t["start"]), float(t["end"])
        if s > prev_end:
            silents.append(SilentInterval(prev_end, s))
        prev_end = max(prev_end, e)
    if video_duration > prev_end:
        silents.append(SilentInterval(prev_end, video_duration))
    return silents


def flow_series(video_path: str) -> tuple[np.ndarray, np.ndarray, float]:
    """샘플 프레임 인접쌍 optical flow magnitude per-frame max 시계열.

    Farneback dense flow → 픽셀별 (dx, dy) 벡터 → magnitude → 프레임 안 max.
    """
    with open_capture(video_path) as cap:
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
                # cv2 stub은 flow=None을 거부 — 빈 ndarray placeholder 전달,
                # OpenCV가 새 buffer alloc.
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

    return (
        np.asarray(curr_times, dtype=np.float64),
        np.asarray(flows, dtype=np.float64),
        duration,
    )


def flow_median_in(
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


def load_audio_or_empty(video_path: str) -> np.ndarray:
    """audio 트랙 없는 영상은 빈 array fallback. 다른 ffmpeg 에러는 그대로 raise."""
    try:
        return whisperx.load_audio(video_path)
    except RuntimeError as e:
        if _FFMPEG_NO_STREAM_MARKER in str(e):
            _log.warning("오디오 트랙 없음 — dead_zone VAD 단계 건너뜀: video=%s", video_path)
            return np.array([], dtype=np.float32)
        raise


async def detect_dead_zone_events(
    video_path: str,
    category: Category,
    *,
    audio: np.ndarray | None = None,
) -> list[DeadZoneEvent]:
    """영상 + 카테고리 → dead zone 이벤트 리스트.

    `audio`가 주어지면 (그래프에서 transcribe 노드가 푸시한 16kHz mono) ffmpeg 재호출 없이
    재사용. 없으면 fallback으로 직접 로드. flow_series(OpenCV)는 audio와 무관해 항상
    별도 스레드로 병렬.
    """
    flow_task = asyncio.to_thread(flow_series, video_path)
    if audio is None:
        audio_task = asyncio.to_thread(load_audio_or_empty, video_path)
        (curr_times, flows, duration), audio = await asyncio.gather(flow_task, audio_task)
    else:
        curr_times, flows, duration = await flow_task
    silent = silent_intervals_from_audio(audio, duration)
    cfg = CATEGORY_CONFIG[category]

    events: list[DeadZoneEvent] = []
    for iv in silent:
        if iv.end - iv.start < cfg.min_duration_sec:
            continue
        median = flow_median_in(curr_times, flows, iv.start, iv.end)
        if median is None or median > cfg.flow_max_threshold:
            continue
        events.append(DeadZoneEvent(start=iv.start, end=iv.end))
    return events
