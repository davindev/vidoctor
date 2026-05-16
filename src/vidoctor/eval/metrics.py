"""검출 vs 라벨 매칭 + 차원별 F1.

차원별 매칭 전략:
- filler: point-in-interval (라벨러 burst vs detector 단발 mismatch)
- cps: ±1s 확장 IoU greedy + kind(too_fast/too_slow) 일치 필수
- dead_zone / gaze / content_gap: IoU greedy 1:1

IoU 임계는 기본 0.3, content_gap만 LLM 정밀도 낮아 0.2로 완화.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vidoctor.eval.labels import GoldenLabel
from vidoctor.graph.state import Dimension

Interval = tuple[float, float]

# match_events default 임계. 평가 파이프라인은 DIM_IOU_THRESHOLD 명시 전달.
IOU_THRESHOLD = 0.3

# 차원별 IoU 임계 — content_gap만 LLM 정밀도 낮아 완화. filler 의도적 부재로
# 잘못 조회 시 KeyError 빠른 실패.
DIM_IOU_THRESHOLD: dict[Dimension, float] = {
    "cps": 0.3,
    "dead_zone": 0.3,
    "gaze": 0.3,
    "content_gap": 0.2,
}

# 라벨러 1초 단위 라운딩 흡수용 ±확장.
FILLER_TOLERANCE_SEC = 1.0
CPS_TOLERANCE_SEC = 1.0


@dataclass
class DimensionMetrics:
    dimension: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    iou_sum: float = 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def temporal_iou_mean(self) -> float:
        return self.iou_sum / self.tp if self.tp > 0 else 0.0


def iou(a: Interval, b: Interval) -> float:
    """두 시간 구간의 Intersection over Union. 겹침 없으면 0.0."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def match_events(
    labels: list[Interval],
    detected: list[Interval],
    iou_threshold: float = IOU_THRESHOLD,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """IoU greedy 1:1 매칭.

    반환: (매칭된 (label_idx, detected_idx, iou) 리스트, 미매칭 label idx, 미매칭 detected idx).
    IoU 큰 쌍부터 채택 — 한 label·detected는 한 번만 매칭.
    """
    candidates: list[tuple[int, int, float]] = []
    for li, lab in enumerate(labels):
        for di, det in enumerate(detected):
            v = iou(lab, det)
            if v >= iou_threshold:
                candidates.append((li, di, v))
    candidates.sort(key=lambda x: -x[2])

    matched_labels: set[int] = set()
    matched_detected: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for li, di, v in candidates:
        if li in matched_labels or di in matched_detected:
            continue
        matched_labels.add(li)
        matched_detected.add(di)
        matches.append((li, di, v))

    unmatched_labels = [li for li in range(len(labels)) if li not in matched_labels]
    unmatched_detected = [di for di in range(len(detected)) if di not in matched_detected]
    return matches, unmatched_labels, unmatched_detected


def match_points_in_intervals(
    labels: list[Interval],
    detected_starts: list[float],
    tolerance: float = 0.0,
) -> tuple[set[int], set[int]]:
    """라벨 구간 안에 detected 시작점이 들어가면 그 라벨 = 매칭.

    burst 라벨 + 단발 detected의 단위 mismatch 처리용. 한 라벨에 detected 여러
    개가 들어가도 모두 "검출 기여"로 묶여 FP에 카운트되지 않음.

    경계는 half-open `[l_start - tolerance, l_end + tolerance)`. tolerance=0이면
    인접 라벨 끝점이 같은 시각이어도 detected는 한 라벨에만 매칭.
    """
    matched_labels: set[int] = set()
    matched_detected: set[int] = set()
    for li, (l_start, l_end) in enumerate(labels):
        for di, det in enumerate(detected_starts):
            if (l_start - tolerance) <= det < (l_end + tolerance):
                matched_labels.add(li)
                matched_detected.add(di)
    return matched_labels, matched_detected


def compute_filler_metrics(
    dim_labels: list[Interval], events: list[Any]
) -> DimensionMetrics:
    """filler 차원 단독 평가 — point-in-interval + ±tolerance."""
    detected_starts = [e.start for e in events]
    matched_labels, matched_detected = match_points_in_intervals(
        dim_labels, detected_starts, tolerance=FILLER_TOLERANCE_SEC
    )
    return DimensionMetrics(
        dimension="filler",
        tp=len(matched_labels),
        fp=len(detected_starts) - len(matched_detected),
        fn=len(dim_labels) - len(matched_labels),
    )


def compute_cps_metrics(
    cps_labels: list[GoldenLabel], events: list[Any]
) -> DimensionMetrics:
    """cps 차원 단독 평가 — 라벨 ±tolerance IoU + kind 일치 greedy 1:1.

    kind가 다른 쌍은 후보 단계에서 제외 → 반대 방향 오분류 TP 차단.
    """
    expanded: list[Interval] = [
        (lbl.start - CPS_TOLERANCE_SEC, lbl.end + CPS_TOLERANCE_SEC)
        for lbl in cps_labels
    ]
    detected: list[Interval] = [(e.start, e.end) for e in events]
    threshold = DIM_IOU_THRESHOLD["cps"]

    # kind 필터 때문에 match_events 재사용 안 함 (cps 전용 파라미터 leak 회피).
    candidates: list[tuple[int, int, float]] = []
    for li, lab in enumerate(expanded):
        for di, det in enumerate(detected):
            if cps_labels[li].kind != events[di].kind:
                continue
            v = iou(lab, det)
            if v >= threshold:
                candidates.append((li, di, v))
    candidates.sort(key=lambda x: -x[2])

    matched_labels: set[int] = set()
    matched_detected: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for li, di, v in candidates:
        if li in matched_labels or di in matched_detected:
            continue
        matched_labels.add(li)
        matched_detected.add(di)
        matches.append((li, di, v))

    return DimensionMetrics(
        dimension="cps",
        tp=len(matches),
        fp=len(detected) - len(matched_detected),
        fn=len(expanded) - len(matched_labels),
        iou_sum=sum(m[2] for m in matches),
    )


def _compute_iou_metrics(
    dim: Dimension, dim_labels: list[Interval], events: list[Any]
) -> DimensionMetrics:
    """IoU greedy 매칭 + DimensionMetrics 패키징 — dead_zone/gaze/content_gap 공용."""
    dim_detected: list[Interval] = [(e.start, e.end) for e in events]
    matches, unmatched_lbl, unmatched_det = match_events(
        dim_labels, dim_detected, DIM_IOU_THRESHOLD[dim]
    )
    return DimensionMetrics(
        dimension=dim,
        tp=len(matches),
        fp=len(unmatched_det),
        fn=len(unmatched_lbl),
        iou_sum=sum(m[2] for m in matches),
    )
