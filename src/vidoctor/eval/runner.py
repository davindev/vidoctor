"""평가 실행 진입점: graph + golden labels → metrics + MLflow run.

사용법:
    uv run python -m vidoctor.eval.runner data/golden/lecture.mp4 \\
        data/golden/lecture_labels.csv lecture
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import cast, get_args

import mlflow

from vidoctor.eval.labels import load_labels
from vidoctor.eval.metrics import EvalReport, compute_metrics
from vidoctor.graph import run_analysis
from vidoctor.graph.state import Category

_EXPERIMENT_NAME = "vidoctor-eval"


def _print_report(video_stem: str, category: str, report: EvalReport) -> None:
    print(f"\n=== {video_stem} ({category}) ===")
    header = f"{'dim':<14} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>6} {'R':>6} {'F1':>6} {'IoU':>6}"
    print(header)
    print("-" * len(header))
    for dim, m in report.per_dimension.items():
        print(
            f"{dim:<14} {m.tp:>4} {m.fp:>4} {m.fn:>4} "
            f"{m.precision:>6.3f} {m.recall:>6.3f} {m.f1:>6.3f} {m.temporal_iou_mean:>6.3f}"
        )
    print(f"\nmacro_f1: {report.macro_f1:.3f}")


def _log_to_mlflow(
    video_path: Path,
    category: Category,
    report: EvalReport,
) -> None:
    mlflow.set_experiment(_EXPERIMENT_NAME)
    # 같은 영상을 여러 번 평가할 때 run name이 충돌하지 않도록 timestamp suffix.
    run_name = f"{video_path.stem}_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"video": video_path.name, "category": category})
        for dim, m in report.per_dimension.items():
            mlflow.log_metrics(
                {
                    f"{dim}_precision": m.precision,
                    f"{dim}_recall": m.recall,
                    f"{dim}_f1": m.f1,
                    f"{dim}_iou_mean": m.temporal_iou_mean,
                    f"{dim}_tp": m.tp,
                    f"{dim}_fp": m.fp,
                    f"{dim}_fn": m.fn,
                }
            )
        mlflow.log_metric("macro_f1", report.macro_f1)


def main() -> None:
    valid_categories = get_args(Category)
    if len(sys.argv) != 4:
        print("usage: runner.py <video_path> <labels_csv> <category>", file=sys.stderr)
        sys.exit(2)

    video_path = Path(sys.argv[1])
    labels_path = Path(sys.argv[2])
    category_arg = sys.argv[3]

    if category_arg not in valid_categories:
        print(
            f"invalid category: {category_arg!r} (expected one of {valid_categories})",
            file=sys.stderr,
        )
        sys.exit(2)
    if not video_path.exists():
        print(f"video not found: {video_path}", file=sys.stderr)
        sys.exit(2)
    if not labels_path.exists():
        print(f"labels not found: {labels_path}", file=sys.stderr)
        sys.exit(2)

    category = cast(Category, category_arg)  # 위에서 valid_categories 검증 완료
    labels = load_labels(labels_path)
    state = asyncio.run(run_analysis(str(video_path), category))
    report = compute_metrics(state, labels)

    _print_report(video_path.stem, category, report)
    _log_to_mlflow(video_path, category, report)


if __name__ == "__main__":
    main()
