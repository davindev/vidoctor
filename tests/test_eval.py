"""평가 시스템 단위 테스트 — IoU / matching / metrics + CSV 로드."""

from __future__ import annotations

from pathlib import Path

import pytest

from vidoctor.eval.labels import load_labels
from vidoctor.eval.metrics import (
    DimensionMetrics,
    iou,
    match_events,
    match_points_in_intervals,
)

# ---------------------------------------------------------------------------
# iou
# ---------------------------------------------------------------------------


def test_iou_full_overlap():
    assert iou((0.0, 10.0), (0.0, 10.0)) == pytest.approx(1.0)


def test_iou_no_overlap():
    assert iou((0.0, 5.0), (10.0, 15.0)) == 0.0


def test_iou_half_overlap():
    # [0,10] vs [5,15] → inter=5, union=15 → 1/3
    assert iou((0.0, 10.0), (5.0, 15.0)) == pytest.approx(1 / 3)


def test_iou_one_inside_other():
    # [0,10] 안에 [3,7] 완전 포함 → inter=4, union=10 → 0.4
    assert iou((0.0, 10.0), (3.0, 7.0)) == pytest.approx(0.4)


def test_iou_touching_intervals_returns_zero():
    # 끝점이 같으면 inter=0, union>0 → 0.0
    assert iou((0.0, 5.0), (5.0, 10.0)) == 0.0


# ---------------------------------------------------------------------------
# match_events — greedy 1:1 IoU 매칭
# ---------------------------------------------------------------------------


def test_match_events_perfect_pairs():
    labels = [(0.0, 5.0), (10.0, 15.0)]
    detected = [(10.0, 15.0), (0.0, 5.0)]
    matches, unmatched_l, unmatched_d = match_events(labels, detected)
    assert len(matches) == 2
    assert unmatched_l == []
    assert unmatched_d == []


def test_match_events_one_label_unmatched():
    labels = [(0.0, 5.0), (50.0, 55.0)]
    detected = [(0.0, 5.0)]
    matches, unmatched_l, unmatched_d = match_events(labels, detected)
    assert len(matches) == 1
    assert unmatched_l == [1]
    assert unmatched_d == []


def test_match_events_one_detected_unmatched():
    labels = [(0.0, 5.0)]
    detected = [(0.0, 5.0), (100.0, 105.0)]
    matches, unmatched_l, unmatched_d = match_events(labels, detected)
    assert len(matches) == 1
    assert unmatched_l == []
    assert unmatched_d == [1]


def test_match_events_below_threshold_skipped():
    # IoU 0.1 → IOU_THRESHOLD(0.3) 미만이라 매칭 안 됨
    labels = [(0.0, 10.0)]
    detected = [(8.0, 18.0)]  # inter=2, union=18 → 0.111
    matches, unmatched_l, unmatched_d = match_events(labels, detected)
    assert matches == []
    assert unmatched_l == [0]
    assert unmatched_d == [0]


def test_match_events_greedy_picks_best_iou_first():
    # detected 1개에 label 2개가 모두 overlap — greedy로 IoU 큰 쪽이 채택
    labels = [(0.0, 10.0), (0.0, 5.0)]
    detected = [(0.0, 5.0)]  # label 1과 IoU 1.0, label 0과 IoU 0.5
    matches, unmatched_l, unmatched_d = match_events(labels, detected)
    assert len(matches) == 1
    assert matches[0][0] == 1  # label idx 1 (더 정확한 매칭)
    assert unmatched_l == [0]


def test_match_events_empty_inputs():
    assert match_events([], []) == ([], [], [])
    assert match_events([(0.0, 5.0)], []) == ([], [0], [])
    assert match_events([], [(0.0, 5.0)]) == ([], [], [0])


# ---------------------------------------------------------------------------
# match_points_in_intervals — filler용 매칭
# ---------------------------------------------------------------------------


def test_match_points_in_intervals_point_inside_label():
    matched_l, matched_d = match_points_in_intervals([(45.0, 68.0)], [66.4])
    assert matched_l == {0}
    assert matched_d == {0}


def test_match_points_in_intervals_point_outside_all_labels():
    matched_l, matched_d = match_points_in_intervals([(45.0, 68.0)], [200.0])
    assert matched_l == set()
    assert matched_d == set()


def test_match_points_in_intervals_multiple_points_in_one_label():
    # burst 라벨 안 단발 3개 → 모두 매칭됨, 라벨 1개 매칭
    matched_l, matched_d = match_points_in_intervals([(45.0, 68.0)], [46.0, 55.0, 66.0])
    assert matched_l == {0}
    assert matched_d == {0, 1, 2}


def test_match_points_in_intervals_boundary_half_open():
    # half-open [start, end): start는 inclusive, end는 exclusive (IoU touching=0과 일관).
    matched_l, matched_d = match_points_in_intervals([(45.0, 68.0)], [45.0, 68.0])
    assert matched_l == {0}  # 시작점 45.0은 매칭
    assert matched_d == {0}  # 시작점만, 끝점 68.0은 매칭 안 됨


def test_match_points_in_intervals_adjacent_labels_no_double_match():
    # 인접 라벨의 끝점=다음 시작점에 detected가 있으면 한 라벨에만 매칭.
    matched_l, _ = match_points_in_intervals([(0.0, 5.0), (5.0, 10.0)], [5.0])
    assert matched_l == {1}  # 시작점 inclusive인 두 번째 라벨


def test_match_points_in_intervals_empty_inputs():
    assert match_points_in_intervals([], []) == (set(), set())
    assert match_points_in_intervals([(0.0, 5.0)], []) == (set(), set())
    assert match_points_in_intervals([], [3.0]) == (set(), set())


# ---------------------------------------------------------------------------
# DimensionMetrics 계산
# ---------------------------------------------------------------------------


def test_dimension_metrics_perfect():
    metrics = DimensionMetrics(dimension="filler", tp=5, fp=0, fn=0, iou_sum=4.5)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.temporal_iou_mean == 0.9


def test_dimension_metrics_only_fp():
    metrics = DimensionMetrics(dimension="filler", tp=0, fp=3, fn=0)
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


def test_dimension_metrics_balanced():
    # P=0.5, R=0.5 → F1=0.5
    metrics = DimensionMetrics(dimension="filler", tp=1, fp=1, fn=1)
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == pytest.approx(0.5)


def test_dimension_metrics_zero_tp_iou_mean_is_zero():
    metrics = DimensionMetrics(dimension="filler", tp=0, fp=2, fn=2)
    assert metrics.temporal_iou_mean == 0.0


# ---------------------------------------------------------------------------
# load_labels — CSV 파싱
# ---------------------------------------------------------------------------


def test_load_labels_real_lecture_csv_if_exists():
    """data/golden/labels/lecture_labels.csv가 있으면 파싱 검증."""
    path = Path("data/golden/labels/lecture_labels.csv")
    if not path.exists():
        pytest.skip(f"missing fixture: {path}")
    labels = load_labels(path)
    assert len(labels) > 0
    valid_dims = {"filler", "cps", "dead_zone", "gaze", "content_gap"}
    for lbl in labels:
        assert lbl.dimension in valid_dims
        assert lbl.start <= lbl.end


def test_load_labels_synthetic_csv(tmp_path: Path):
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "start,end,dimension,kind,note\n"
        "1.0,2.0,filler,,음\n"
        "10.0,15.0,cps,too_fast,속사포\n"
        ",,,,\n"  # 빈 행 — skip
        "20.0,40.0,dead_zone,,\n",
        encoding="utf-8",
    )
    labels = load_labels(csv_path)
    assert len(labels) == 3
    assert labels[0].dimension == "filler"
    assert labels[1].kind == "too_fast"
    assert labels[2].kind is None


def test_load_labels_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_labels(tmp_path / "nope.csv")


def test_load_labels_raises_on_missing_required_column(tmp_path: Path):
    # 비싼 ASR 호출 전에 헤더 검증으로 abort하는지 — fail-fast 회귀 가드.
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("start,end,note\n1.0,2.0,foo\n", encoding="utf-8")
    with pytest.raises(ValueError, match="필수 컬럼이 빠졌습니다"):
        load_labels(csv_path)


def test_load_labels_raises_on_unknown_column(tmp_path: Path):
    # 오타·미지의 컬럼은 즉시 reject — 라벨러 실수 조기 발견.
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(
        "start,end,dimension,extra\n1.0,2.0,filler,x\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="알 수 없는 컬럼"):
        load_labels(csv_path)
