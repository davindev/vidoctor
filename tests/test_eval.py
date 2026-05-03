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
)
from vidoctor.graph.state import (
    AnalysisState,
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


def test_compute_metrics_partial_overlap_counted_as_tp():
    state = _state(fillers=[FillerEvent(start=0.5, end=2.5, text="음")])
    labels = [GoldenLabel(start=1.0, end=2.0, dimension="filler")]
    report = compute_metrics(state, labels)
    m = report.per_dimension["filler"]
    assert m.tp == 1
    assert m.iou_sum == pytest.approx(0.5)  # inter=1, union=2 → 0.5


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
    # filler P=R=F1=1.0, gaze P=R=F1=0.0 → macro 0.5
    state = _state(
        fillers=[FillerEvent(start=1.0, end=2.0, text="음")],
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
    """data/golden/lecture_labels.csv가 있으면 파싱 검증."""
    path = Path("data/golden/lecture_labels.csv")
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
        "start,end,dimension,severity,kind,note\n"
        "1.0,2.0,filler,mid,,음\n"
        "10.0,15.0,cps,high,too_fast,속사포\n"
        ",,,,,\n"  # 빈 행 — skip
        "20.0,40.0,dead_zone,mid,,\n",
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
