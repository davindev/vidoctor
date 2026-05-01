"""Dead zone 검출 테스트.

순수 함수(_silent_intervals, _intersect_intervals) 단위 + 합성 영상으로 통합 검증.
"""

from pathlib import Path

import numpy as np
import pytest

from tests._helpers import _w, write_video
from vidoctor.vision.dead_zone import (
    _detect_dead_zone_sync,
    _intersect_intervals,
    _Interval,
    _silent_intervals,
)

# ---------------------------------------------------------------------------
# _silent_intervals
# ---------------------------------------------------------------------------


def test_silent_intervals_empty_transcript_full_video():
    silent = _silent_intervals([], 10.0)
    assert silent == [_Interval(0.0, 10.0)]


def test_silent_intervals_empty_transcript_zero_duration():
    assert _silent_intervals([], 0.0) == []


def test_silent_intervals_short_gap_not_detected():
    words = [_w("a", 0.0, 0.5), _w("b", 0.7, 1.0)]
    assert _silent_intervals(words, 1.0) == []


def test_silent_intervals_long_gap_detected():
    words = [_w("a", 0.0, 0.5), _w("b", 5.0, 5.5)]
    silent = _silent_intervals(words, 6.0)
    assert _Interval(0.5, 5.0) in silent


def test_silent_intervals_leading_silence():
    words = [_w("a", 5.0, 5.5)]
    silent = _silent_intervals(words, 6.0)
    assert _Interval(0.0, 5.0) in silent


def test_silent_intervals_trailing_silence():
    words = [_w("a", 0.0, 0.5)]
    silent = _silent_intervals(words, 10.0)
    assert _Interval(0.5, 10.0) in silent


# ---------------------------------------------------------------------------
# _intersect_intervals
# ---------------------------------------------------------------------------


def test_intersect_no_overlap():
    a = [_Interval(0.0, 5.0)]
    b = [_Interval(10.0, 15.0)]
    assert _intersect_intervals(a, b) == []


def test_intersect_full_overlap():
    a = [_Interval(0.0, 10.0)]
    b = [_Interval(0.0, 10.0)]
    assert _intersect_intervals(a, b) == [_Interval(0.0, 10.0)]


def test_intersect_partial_overlap():
    a = [_Interval(0.0, 10.0), _Interval(20.0, 30.0)]
    b = [_Interval(5.0, 25.0)]
    result = _intersect_intervals(a, b)
    assert _Interval(5.0, 10.0) in result
    assert _Interval(20.0, 25.0) in result


def test_intersect_empty():
    assert _intersect_intervals([], [_Interval(0.0, 1.0)]) == []
    assert _intersect_intervals([_Interval(0.0, 1.0)], []) == []


# ---------------------------------------------------------------------------
# 합성 영상 통합 — _detect_dead_zone_sync
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
    """0~2s noise + 2~10s 정적 영상 (총 10s)."""
    video = tmp_path / "test.mp4"
    _make_test_video(video, duration_sec=10.0, static_from=2.0)
    assert video.exists() and video.stat().st_size > 0
    return video


def test_detect_lecture_threshold_skips_short_static(synthetic_video: Path):
    # 강의 임계 20s, 합성 영상은 8s 정적 → 검출 안 함
    events = _detect_dead_zone_sync(str(synthetic_video), [], "lecture")
    assert events == []


def test_detect_vlog_threshold_finds_static(synthetic_video: Path):
    # 브이로그 임계 5s, 8s 정적 + 무발화 → 검출
    events = _detect_dead_zone_sync(str(synthetic_video), [], "vlog")
    assert len(events) >= 1
    assert events[0].end - events[0].start >= 5.0


def test_detect_speech_during_static_blocks_detection(synthetic_video: Path):
    # 정적 구간(2~10s)에 발화가 채워져 있으면 dead zone 아님
    transcript = [_w("말함", t, t + 0.4) for t in np.arange(2.0, 10.0, 0.5)]
    events = _detect_dead_zone_sync(str(synthetic_video), transcript, "vlog")
    assert events == []
