"""검출 vs 라벨 overlap matching + 차원별 F1.

알고리즘:
1. 차원별로 detected와 label을 분리
2. 각 (label, detected) 쌍의 IoU 계산 (시간 구간 IoU)
3. greedy 1:1 매칭 — IoU 가장 큰 쌍부터 채택, label·detected 한 번씩만
4. IoU >= IOU_THRESHOLD 미달은 매칭 안 함
5. 매칭 = TP, 미매칭 detected = FP, 미매칭 label = FN
6. precision/recall/F1 + temporal IoU mean

severity 가중은 v1.1 (현재 모든 라벨이 mid 통일이라 가중 무의미). Cohen's κ도
라벨러 ≥ 2명일 때 inter-rater agreement 측정용이라 v1.1로 연기.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vidoctor.eval.labels import GoldenLabel
from vidoctor.graph.state import DIM_TO_STATE_FIELD, AnalysisState

# overlap-based matching 임계. 라벨링·detector 양쪽 시간 정밀도가 ±수 초 흔들리는 걸 받아주되
# 잘못된 우연 매칭은 거름. 골든셋 분포 확인 후 조정 가능.
#
# v1.0 통일 IoU 매칭의 알려진 한계: filler 라벨러가 burst를 한 row(20초+)로 묶고 detector는
# 단발 어휘별(0.1~0.5초) 출력 → 단위 mismatch로 IoU 매칭 어려움. 차원별 매칭 전략(filler는
# point-in-interval 등)은 별도 변경에서 도입 예정.
IOU_THRESHOLD = 0.3

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
    """greedy 1:1 매칭.

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


def compute_metrics(
    state: AnalysisState,
    labels: list[GoldenLabel],
    iou_threshold: float = IOU_THRESHOLD,
) -> EvalReport:
    """차원별 F1 + macro F1.

    라벨도 검출도 0인 차원은 평가 대상에서 제외 — 카테고리별 비활성 차원 자동 처리.
    """
    report = EvalReport()

    for dim, field_name in DIM_TO_STATE_FIELD.items():
        dim_labels: list[Interval] = [
            (lbl.start, lbl.end) for lbl in labels if lbl.dimension == dim
        ]
        events: list[Any] = list(state.get(field_name, []) or [])  # type: ignore[literal-required]
        dim_detected: list[Interval] = [(e.start, e.end) for e in events]

        if not dim_labels and not dim_detected:
            continue

        matches, unmatched_lbl, unmatched_det = match_events(
            dim_labels, dim_detected, iou_threshold
        )
        report.per_dimension[dim] = DimensionMetrics(
            dimension=dim,
            tp=len(matches),
            fp=len(unmatched_det),
            fn=len(unmatched_lbl),
            iou_sum=sum(m[2] for m in matches),
        )

    return report
