"""Gaze 검출 테스트.

순수 함수(_label_direction / _samples_to_events / _normalize_pose_angle) 단위 +
정면 응시 6점 합성 입력으로 _solve_head_pose sanity check + 카테고리 가드.
실 영상은 VIDOCTOR_RUN_INTEGRATION=1에서만.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import numpy as np
import pytest

from vidoctor.vision.gaze import (
    PITCH_THRESHOLD_DEG,
    YAW_THRESHOLD_DEG,
    _is_off,
    _label_direction,
    _normalize_pose_angle,
    _PoseSample,
    _samples_to_events,
    _solve_head_pose,
    _subtract_baseline,
    detect_gaze_events,
)

# ---------------------------------------------------------------------------
# _normalize_pose_angle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0.0, 0.0),
        (45.0, 45.0),
        (-45.0, -45.0),
        (170.0, 10.0),
        (-170.0, -10.0),
        (190.0, -10.0),
        (-190.0, 10.0),
    ],
)
def test_normalize_pose_angle(raw: float, expected: float):
    assert _normalize_pose_angle(raw) == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# _label_direction
# ---------------------------------------------------------------------------


def test_label_direction_front_when_within_threshold():
    assert _label_direction(0.0, 0.0) == "front"
    assert _label_direction(YAW_THRESHOLD_DEG - 1, PITCH_THRESHOLD_DEG - 1) == "front"


def test_label_direction_horizontal_only():
    assert _label_direction(YAW_THRESHOLD_DEG + 5, 0.0) == "right"
    assert _label_direction(-(YAW_THRESHOLD_DEG + 5), 0.0) == "left"


def test_label_direction_vertical_only():
    assert _label_direction(0.0, PITCH_THRESHOLD_DEG + 5) == "down"
    assert _label_direction(0.0, -(PITCH_THRESHOLD_DEG + 5)) == "up"


def test_label_direction_diagonal_combines():
    # 노트북 응시 = right_down 또는 left_down 패턴 — 두 축 모두 표기.
    assert _label_direction(YAW_THRESHOLD_DEG + 5, PITCH_THRESHOLD_DEG + 5) == "right_down"
    assert _label_direction(-(YAW_THRESHOLD_DEG + 5), -(PITCH_THRESHOLD_DEG + 5)) == "left_up"


# ---------------------------------------------------------------------------
# _is_off / _samples_to_events
# ---------------------------------------------------------------------------


_OFF_YAW = YAW_THRESHOLD_DEG + 5
_OFF_PITCH = PITCH_THRESHOLD_DEG + 5


def _front(t: float) -> _PoseSample:
    return _PoseSample(t=t, yaw=0.0, pitch=0.0)


def _off_right(t: float) -> _PoseSample:
    return _PoseSample(t=t, yaw=_OFF_YAW, pitch=0.0)


def _off_left(t: float) -> _PoseSample:
    return _PoseSample(t=t, yaw=-_OFF_YAW, pitch=0.0)


def _off_down(t: float) -> _PoseSample:
    return _PoseSample(t=t, yaw=0.0, pitch=_OFF_PITCH)


def test_is_off_within_threshold_returns_false():
    assert _is_off(_front(0.0)) is False


def test_is_off_yaw_or_pitch_exceeds_threshold():
    assert _is_off(_off_right(0.0)) is True
    assert _is_off(_off_down(0.0)) is True


def test_samples_empty():
    assert _samples_to_events([]) == []


def test_samples_all_front_no_event():
    assert _samples_to_events([_front(0.0), _front(1.0), _front(2.0)]) == []


def test_samples_short_off_below_min_duration_skipped():
    # 이탈 지속 0.2s가 MIN_DURATION_SEC 미만 → 이벤트 없음.
    samples = [_front(0.0), _off_right(0.2), _off_right(0.4), _front(0.6), _front(1.2)]
    assert _samples_to_events(samples) == []


def test_samples_long_off_emits_event_with_direction():
    samples = [_off_down(t) for t in (0.0, 0.5, 1.0, 1.5, 2.0)] + [_front(2.5), _front(3.0)]
    events = _samples_to_events(samples)
    assert len(events) == 1
    assert events[0].direction == "down"
    assert events[0].start == 0.0
    assert events[0].end == 2.0


def test_samples_short_front_blip_merges_within_gap():
    # 중간 한 프레임만 정면(0.8s 갭) → MERGE_GAP_SEC 이내라 같은 이벤트로 묶임.
    samples = [
        _off_left(0.0),
        _off_left(0.4),
        _off_left(0.8),
        _off_left(1.2),
        _front(1.6),
        _off_left(2.0),
        _off_left(2.4),
        _off_left(2.8),
    ]
    events = _samples_to_events(samples)
    assert len(events) == 1
    assert events[0].start == 0.0
    assert events[0].end == 2.8


def test_samples_long_front_gap_splits_events():
    samples = [
        _off_right(0.0),
        _off_right(0.5),
        _off_right(1.0),
        _front(1.5),
        _front(2.5),
        _off_left(3.5),
        _off_left(4.0),
        _off_left(4.5),
    ]
    events = _samples_to_events(samples)
    assert len(events) == 2
    assert events[0].direction == "right"
    assert events[1].direction == "left"


# ---------------------------------------------------------------------------
# _subtract_baseline
# ---------------------------------------------------------------------------


def test_subtract_baseline_empty_returns_empty():
    assert _subtract_baseline([]) == ([], 0.0, 0.0)


def test_subtract_baseline_centers_yaw_pitch_on_median():
    samples = [
        _PoseSample(t=0.0, yaw=2.0, pitch=-12.0),
        _PoseSample(t=1.0, yaw=3.0, pitch=-10.0),
        _PoseSample(t=2.0, yaw=4.0, pitch=-9.0),
        _PoseSample(t=3.0, yaw=5.0, pitch=-8.0),
        _PoseSample(t=4.0, yaw=6.0, pitch=-6.0),
    ]
    out, by, bp = _subtract_baseline(samples)
    assert (by, bp) == (4.0, -9.0)
    assert [s.yaw for s in out] == [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert [s.pitch for s in out] == [-3.0, -1.0, 0.0, 1.0, 3.0]


def test_subtract_baseline_robust_to_outlier():
    # 짧은 시선 이탈(1개 큰 yaw)이 baseline 추정을 오염시키지 않아야 — median 사용.
    samples = [_PoseSample(t=float(i), yaw=0.0, pitch=0.0) for i in range(9)]
    samples.append(_PoseSample(t=9.0, yaw=80.0, pitch=0.0))
    out, by, _ = _subtract_baseline(samples)
    assert by == 0.0
    assert out[0].yaw == 0.0
    assert out[-1].yaw == 80.0


# ---------------------------------------------------------------------------
# _solve_head_pose sanity (정면 응시 가정한 합성 입력)
# ---------------------------------------------------------------------------


def test_solve_head_pose_front_facing_returns_small_yaw_pitch():
    # 정면 응시 가정: 6개 model 점을 그대로 image plane에 평행 투영한 좌표.
    width, height = 640, 480
    cx, cy = width / 2.0, height / 2.0
    points_2d = np.array(
        [
            (cx, cy),
            (cx, cy + 60.0),
            (cx - 40.0, cy - 30.0),
            (cx + 40.0, cy - 30.0),
            (cx - 28.0, cy + 28.0),
            (cx + 28.0, cy + 28.0),
        ],
        dtype=np.float64,
    )
    pose = _solve_head_pose(points_2d, width, height)
    assert pose is not None
    yaw, pitch = pose
    assert abs(yaw) < YAW_THRESHOLD_DEG
    assert abs(pitch) < PITCH_THRESHOLD_DEG


# ---------------------------------------------------------------------------
# 실 영상 통합 — VIDOCTOR_RUN_INTEGRATION=1 일 때만
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("VIDOCTOR_RUN_INTEGRATION") != "1",
    reason="set VIDOCTOR_RUN_INTEGRATION=1 to run real-video gaze integration",
)
def test_detect_gaze_on_lecture_video_runs_without_crash():
    video = Path("data/golden/lecture.mp4")
    if not video.exists():
        pytest.skip(f"missing fixture: {video}")
    events = asyncio.run(detect_gaze_events(str(video)))
    # 강의 영상은 화자 정면 응시가 기본이라 0개여도 정상. 크래시·타입 무결성만 보증.
    assert isinstance(events, list)
    for e in events:
        assert e.start <= e.end
        assert e.direction
