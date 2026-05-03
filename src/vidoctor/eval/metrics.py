"""검출 vs 라벨 매칭 + 차원별 F1.

차원별 매칭 전략:
- **filler**: point-in-interval. 라벨러는 burst 구간으로 묶고 detector는 단발 어휘별
  (0.1~0.5초)이라 IoU로는 단위 mismatch. 라벨 구간 안에 detected 시작점이 1개 이상
  들어가면 그 라벨 = TP. burst 안 detected는 모두 "검출 기여"로 묶여 FP에 카운트 X.
- **나머지** (cps / dead_zone / gaze / content_gap): IoU greedy 1:1 매칭.

IoU 임계도 차원별:
- content_gap은 LLM 출력의 시간 정밀도가 낮아 0.2로 완화
- 그 외는 기본 0.3

severity 가중·Cohen's κ는 v1.1 (현재 모든 라벨이 mid 통일).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vidoctor.eval.labels import GoldenLabel
from vidoctor.graph.state import DIM_TO_STATE_FIELD, AnalysisState, Dimension

# 기본 IoU 임계. 라벨링·detector 양쪽 시간 정밀도가 ±수 초 흔들리는 걸 받아주되
# 잘못된 우연 매칭은 거름.
IOU_THRESHOLD = 0.3

# 매칭 전략이 다른 차원. 라벨러는 burst, detector는 단발이라 IoU 대신 point-in-interval.
POINT_DIMENSIONS: frozenset[Dimension] = frozenset({"filler"})

# IoU 매칭 차원의 임계. content_gap은 LLM 출력의 시간 정밀도가 떨어져 완화.
# IoU 매칭 차원만 명시 — POINT_DIMENSIONS 차원은 들어가지 않음 (직접 조회 시 KeyError로 빠른 실패).
DIM_IOU_THRESHOLD: dict[Dimension, float] = {
    "cps": 0.3,
    "dead_zone": 0.3,
    "gaze": 0.3,
    "content_gap": 0.2,
}

Interval = tuple[float, float]


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


@dataclass
class EvalReport:
    per_dimension: dict[str, DimensionMetrics] = field(default_factory=dict)

    @property
    def macro_f1(self) -> float:
        """평가 대상 차원의 F1 평균. 라벨도 검출도 0이라 skip된 차원은 평균 계산에서도 제외."""
        if not self.per_dimension:
            return 0.0
        return sum(m.f1 for m in self.per_dimension.values()) / len(self.per_dimension)


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
) -> tuple[set[int], set[int]]:
    """라벨 구간 안에 detected 시작점이 1개 이상 들어가면 그 라벨 = 매칭.

    burst 라벨(예: 45~68s) + 단발 detected(예: 66.4s)의 mismatch를 처리하기 위함.
    한 라벨에 detected 여러 개가 들어가도 그 detected들은 "검출 기여"로 모두 매칭됨 처리
    (FP에 안 들어감) — detector가 잘게 쪼갠 게 잘못이 아니므로.

    경계 정책: half-open `[l_start, l_end)`. IoU의 touching=0 컨벤션과 일관 — 인접
    라벨 끝점이 같은 시각이면 detected는 한 라벨에만 매칭.

    반환: (매칭된 label idx 집합, 매칭된 detected idx 집합).
    """
    matched_labels: set[int] = set()
    matched_detected: set[int] = set()
    for li, (l_start, l_end) in enumerate(labels):
        for di, det in enumerate(detected_starts):
            if l_start <= det < l_end:
                matched_labels.add(li)
                matched_detected.add(di)
    return matched_labels, matched_detected


def _compute_filler_metrics(
    dim: Dimension, dim_labels: list[Interval], events: list[Any]
) -> DimensionMetrics:
    """filler는 point-in-interval 매칭. iou_sum은 의미 없으므로 0으로 유지."""
    detected_starts = [e.start for e in events]
    matched_labels, matched_detected = match_points_in_intervals(
        dim_labels, detected_starts
    )
    return DimensionMetrics(
        dimension=dim,
        tp=len(matched_labels),
        fp=len(detected_starts) - len(matched_detected),
        fn=len(dim_labels) - len(matched_labels),
    )


def _compute_iou_metrics(
    dim: Dimension, dim_labels: list[Interval], events: list[Any]
) -> DimensionMetrics:
    """fallback 없이 직접 조회 — POINT_DIMENSIONS 차원이 잘못 분기되면 KeyError로 빠른 실패."""
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


def compute_metrics(state: AnalysisState, labels: list[GoldenLabel]) -> EvalReport:
    """차원별 F1 + macro F1. 차원별 매칭 전략 분기.

    라벨도 검출도 0인 차원은 평가 대상에서 제외 — 카테고리별 비활성 차원 자동 처리.
    """
    report = EvalReport()

    for dim, field_name in DIM_TO_STATE_FIELD.items():
        dim_labels: list[Interval] = [
            (lbl.start, lbl.end) for lbl in labels if lbl.dimension == dim
        ]
        events: list[Any] = list(state.get(field_name, []) or [])  # type: ignore[literal-required]

        if not dim_labels and not events:
            continue

        if dim in POINT_DIMENSIONS:
            report.per_dimension[dim] = _compute_filler_metrics(dim, dim_labels, events)
        else:
            report.per_dimension[dim] = _compute_iou_metrics(dim, dim_labels, events)

    return report
