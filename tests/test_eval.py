"""평가 시스템 단위 테스트 — IoU / matching / metrics + CSV 로드."""

from __future__ import annotations

from pathlib import Path

import pytest

from vidoctor.eval.labels import GoldenLabel, load_labels
from vidoctor.eval.metrics import (
    DimensionMetrics,
    compute_metrics,
    iou,
    match_events,
    match_points_in_intervals,
)
from vidoctor.graph.state import (
    AnalysisState,
    ContentGapEvent,
    DeadZoneEvent,
    FillerEvent,
    GazeEvent,
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
    m = DimensionMetrics(dimension="filler", tp=5, fp=0, fn=0, iou_sum=4.5)
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.temporal_iou_mean == 0.9


def test_dimension_metrics_only_fp():
    m = DimensionMetrics(dimension="filler", tp=0, fp=3, fn=0)
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0


def test_dimension_metrics_balanced():
    # P=0.5, R=0.5 → F1=0.5
    m = DimensionMetrics(dimension="filler", tp=1, fp=1, fn=1)
    assert m.precision == 0.5
    assert m.recall == 0.5
    assert m.f1 == pytest.approx(0.5)


def test_dimension_metrics_zero_tp_iou_mean_is_zero():
    m = DimensionMetrics(dimension="filler", tp=0, fp=2, fn=2)
    assert m.temporal_iou_mean == 0.0


# ---------------------------------------------------------------------------
# compute_metrics — graph state + labels 통합
# ---------------------------------------------------------------------------


def _state(**fields: object) -> AnalysisState:
    base: dict[str, object] = {"video_path": "x", "category": "lecture"}
    base.update(fields)
    return base  # type: ignore[return-value]


def test_compute_metrics_skips_dimensions_with_no_label_and_no_detection():
    state = _state(fillers=[FillerEvent(start=1.0, end=2.0, text="음")])
    labels = [GoldenLabel(start=1.0, end=2.0, dimension="filler")]
    report = compute_metrics(state, labels)
    # filler만 평가, 나머지 차원은 라벨/검출 없어 skip
    assert set(report.per_dimension.keys()) == {"filler"}
    assert report.per_dimension["filler"].tp == 1
    assert report.macro_f1 == 1.0


def test_compute_metrics_filler_uses_point_in_interval():
    # filler는 IoU가 아니라 detected 시작점이 라벨 구간 안에 있는지로 매칭.
    # detected start=66.4가 label 45-68 안 → TP. iou_sum은 의미 없어 0.
    state = _state(fillers=[FillerEvent(start=66.4, end=66.5, text="그")])
    labels = [GoldenLabel(start=45.0, end=68.0, dimension="filler")]
    report = compute_metrics(state, labels)
    m = report.per_dimension["filler"]
    assert m.tp == 1
    assert m.fp == 0
    assert m.fn == 0
    assert m.iou_sum == 0.0  # filler 분기는 IoU 안 씀


def test_compute_metrics_filler_burst_with_multiple_detections_no_fp():
    # 한 burst 라벨 안에 단발 detected 여러 개 → 모두 "검출 기여"로 묶여 FP 0, TP 1.
    state = _state(
        fillers=[
            FillerEvent(start=46.0, end=46.2, text="음"),
            FillerEvent(start=55.0, end=55.3, text="어"),
            FillerEvent(start=66.0, end=66.4, text="그"),
        ]
    )
    labels = [GoldenLabel(start=45.0, end=68.0, dimension="filler")]
    report = compute_metrics(state, labels)
    m = report.per_dimension["filler"]
    assert m.tp == 1
    assert m.fp == 0
    assert m.fn == 0


def test_compute_metrics_filler_outside_label_counts_as_fp():
    state = _state(
        fillers=[
            FillerEvent(start=46.0, end=46.2, text="음"),  # label 45-68 안 → TP
            FillerEvent(start=200.0, end=200.3, text="좀"),  # label 밖 → FP
        ]
    )
    labels = [GoldenLabel(start=45.0, end=68.0, dimension="filler")]
    report = compute_metrics(state, labels)
    m = report.per_dimension["filler"]
    assert m.tp == 1
    assert m.fp == 1
    assert m.fn == 0


def test_compute_metrics_content_gap_uses_relaxed_iou_threshold():
    # IoU 0.27은 0.3 미달이지만 content_gap 임계 0.2엔 통과 → TP.
    state = _state(
        content_gaps=[ContentGapEvent(start=75.0, end=105.0, description="x")]
    )
    labels = [GoldenLabel(start=94.0, end=102.0, dimension="content_gap")]
    report = compute_metrics(state, labels)
    m = report.per_dimension["content_gap"]
    assert m.tp == 1
    assert m.fn == 0


def test_compute_metrics_unmatched_label_counts_as_fn():
    state = _state(fillers=[])
    labels = [GoldenLabel(start=1.0, end=2.0, dimension="filler")]
    report = compute_metrics(state, labels)
    m = report.per_dimension["filler"]
    assert m.tp == 0
    assert m.fn == 1
    assert m.fp == 0


def test_compute_metrics_unmatched_detection_counts_as_fp():
    state = _state(fillers=[FillerEvent(start=1.0, end=2.0, text="음")])
    labels: list[GoldenLabel] = []
    # filler 검출 1개, label 0개 → label/detected 둘 다 비지 않은 차원이라 평가 대상
    report = compute_metrics(state, labels)
    assert report.per_dimension["filler"].fp == 1


def test_compute_metrics_macro_f1_averages_per_dimension():
    # filler P=R=F1=1.0 (point-in-interval), gaze P=R=F1=0.0 (IoU 0) → macro 0.5
    state = _state(
        fillers=[FillerEvent(start=1.5, end=1.7, text="음")],
        gaze_issues=[GazeEvent(start=10.0, end=11.0, direction="down")],
    )
    labels = [
        GoldenLabel(start=1.0, end=2.0, dimension="filler"),
        GoldenLabel(start=50.0, end=55.0, dimension="gaze"),
    ]
    report = compute_metrics(state, labels)
    assert report.per_dimension["filler"].f1 == 1.0
    assert report.per_dimension["gaze"].f1 == 0.0
    assert report.macro_f1 == pytest.approx(0.5)


def test_compute_metrics_multi_dimension_isolated():
    """다른 차원의 라벨·검출이 서로 매칭되지 않아야 함."""
    state = _state(
        fillers=[FillerEvent(start=1.0, end=2.0, text="음")],
        dead_zones=[DeadZoneEvent(start=1.0, end=2.0)],
    )
    labels = [GoldenLabel(start=1.0, end=2.0, dimension="filler")]
    # dead_zone에는 라벨이 없는데 검출이 있어 FP 1
    report = compute_metrics(state, labels)
    assert report.per_dimension["filler"].tp == 1
    assert report.per_dimension["dead_zone"].fp == 1
    assert report.per_dimension["dead_zone"].fn == 0


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
