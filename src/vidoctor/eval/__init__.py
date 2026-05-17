"""골든셋 평가 — 라벨 로더 + 차원별 매칭/메트릭."""

from vidoctor.eval.labels import GoldenLabel, load_labels
from vidoctor.eval.metrics import (
    DimensionMetrics,
    iou,
    match_events,
    match_points_in_intervals,
)

__all__ = [
    "DimensionMetrics",
    "GoldenLabel",
    "iou",
    "load_labels",
    "match_events",
    "match_points_in_intervals",
]
