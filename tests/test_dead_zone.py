"""Dead zone 검출 테스트.

VAD silent + Optical flow 단위 + 합성 영상 통합 검증.
"""

from pathlib import Path

import numpy as np
import pytest

from tests._helpers import write_video
from vidoctor.graph.state import Category
from vidoctor.vision.dead_zone import (
    SilentInterval,
    detect_dead_zone_events,
    flow_median_in,
    silent_intervals_from_audio,
)

# ---------------------------------------------------------------------------
# silent_intervals_from_audio (Silero VAD wrapper)
# ---------------------------------------------------------------------------


def test_silent_intervals_empty_audio_full_video():
    silent = silent_intervals_from_audio(np.array([], dtype=np.float32), 10.0)
    assert silent == [SilentInterval(0.0, 10.0)]


def test_silent_intervals_empty_audio_zero_duration():
    assert silent_intervals_from_audio(np.array([], dtype=np.float32), 0.0) == []


def test_silent_intervals_pure_silence_audio():
    silent_audio = np.zeros(16000 * 5, dtype=np.float32)
    silent = silent_intervals_from_audio(silent_audio, 5.0)
    assert silent == [SilentInterval(0.0, 5.0)]


def test_flow_median_in_exact_boundary_inclusive():
    # 경계 시각이 정확히 일치할 때 양 끝이 윈도우에 포함되는지 — half-open 여부 회귀 가드.
    curr = np.array([2.0, 3.0, 4.0])
    flows = np.array([0.1, 0.5, 0.9])
    # 2.0 ~ 4.0 윈도우. 양 끝 포함 시 median 0.5, 한쪽만이면 0.3 또는 0.7
    assert flow_median_in(curr, flows, 2.0, 4.0) == pytest.approx(0.5)


def test_flow_median_in_single_sample_window():
    # 윈도우 안에 샘플 1개만 → median = 그 값.
    curr = np.array([1.0, 3.0, 5.0])
    flows = np.array([0.1, 0.7, 0.9])
    assert flow_median_in(curr, flows, 2.5, 3.5) == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# flow_median_in
# ---------------------------------------------------------------------------


def test_flow_median_in_returns_window_median():
    curr = np.array([1.0, 2.0, 3.0, 4.0])
    flows = np.array([0.05, 0.1, 0.5, 0.2])
    # 2~4s window: 0.1, 0.5, 0.2 → median 0.2
    assert flow_median_in(curr, flows, 2.0, 4.0) == pytest.approx(0.2)


def test_flow_median_in_no_samples_returns_none():
    curr = np.array([10.0, 20.0])
    flows = np.array([0.01, 0.01])
    # 0~5s 안 샘플 없음 → None (caller가 명시적으로 가드)
    assert flow_median_in(curr, flows, 0.0, 5.0) is None


# ---------------------------------------------------------------------------
# 합성 영상 통합 — detect_dead_zone_events
# ---------------------------------------------------------------------------


def _make_test_video(path: Path, duration_sec: float = 10.0, static_from: float = 2.0) -> None:
    """`static_from`초 이후로 정적 프레임, 그 전엔 random noise."""
    rng = np.random.default_rng(42)
    static_frame = np.ones((240, 320, 3), dtype=np.uint8) * 128

    def frame_fn(_i: int, t: float) -> np.ndarray:
        if t >= static_from:
            return static_frame
        return rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)

    write_video(path, duration_sec, frame_fn)


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """0~2s noise + 2~10s 정적 영상. audio 없음 → 영상 전체 무발화 처리."""
    video = tmp_path / "test.mp4"
    _make_test_video(video, duration_sec=10.0, static_from=2.0)
    assert video.exists() and video.stat().st_size > 0
    return video


@pytest.mark.parametrize("category", ["lecture", "vlog", "other"])
async def test_detect_finds_static_with_silent_audio(synthetic_video: Path, category: Category):
    # 합성 영상 무성 → 영상 전체 무발화. 정적 부분 flow median ≈ 0 → 통과.
    events = await detect_dead_zone_events(str(synthetic_video), category)
    assert len(events) >= 1


async def test_detect_noisy_video_blocked_by_flow_gate(tmp_path: Path):
    # 영상 전체 noise (정적 구간 없음) → flow median 큼 → 차단.
    video = tmp_path / "noisy.mp4"
    _make_test_video(video, duration_sec=10.0, static_from=999.0)
    events = await detect_dead_zone_events(str(video), "vlog")
    assert events == []


@pytest.mark.parametrize("category", ["lecture", "vlog", "other"])
async def test_detect_short_video_below_min_duration(tmp_path: Path, category: Category):
    # 영상 4s — min_duration 5s 미만이라 어떤 카테고리도 검출 안 함.
    video = tmp_path / "short.mp4"
    _make_test_video(video, duration_sec=4.0, static_from=0.0)
    events = await detect_dead_zone_events(str(video), category)
    assert events == []
