"""filler 차원 단독 평가 — P/R/F1 + 라벨별 ASR 토큰 dump + MLflow 기록.

사용법:
    uv run python scripts/filler_eval.py data/golden/inputs/lecture.mp4 \\
        data/golden/labels/lecture_labels.csv --run-name baseline_lecture
"""

from __future__ import annotations

import logging

from vidoctor.audio.filler import detect_filler_events
from vidoctor.eval._script_lib import (
    build_eval_parser,
    configure_eval_logging,
    eval_dump_path,
    experiment_name,
    filter_labels_by_dim,
    load_or_transcribe,
    log_mlflow_run,
    metrics_to_dict,
    write_eval_dump,
)
from vidoctor.eval.labels import load_labels
from vidoctor.eval.metrics import compute_filler_metrics
from vidoctor.graph.state import Word

_log = logging.getLogger(__name__)
_DIMENSION = "filler"


def _label_time_tokens(
    intervals: list[tuple[float, float]], words: list[Word]
) -> list[dict]:
    """각 라벨 구간에 들어간 ASR 토큰 목록 — 사전·임계 튜닝 진단용."""
    return [
        {
            "label": [ls, le],
            "tokens": [
                {"start": w.start, "end": w.end, "text": w.text}
                for w in words
                if ls <= w.start < le
            ],
        }
        for ls, le in intervals
    ]


def main() -> None:
    parser = build_eval_parser("filler P/R/F1 + MLflow logging")
    args = parser.parse_args()
    configure_eval_logging(args.run_name)

    words = load_or_transcribe(args.video_path, args.no_cache)
    events = detect_filler_events(words)

    labels = load_labels(args.labels_csv)
    filler_labels = [
        (lbl.start, lbl.end) for lbl in filter_labels_by_dim(labels, _DIMENSION)
    ]

    metrics = metrics_to_dict(
        compute_filler_metrics(filler_labels, events), include_iou=False
    )
    _log.info(
        "filler: TP=%d FP=%d FN=%d P=%.3f R=%.3f F1=%.3f",
        metrics["tp"], metrics["fp"], metrics["fn"],
        metrics["precision"], metrics["recall"], metrics["f1"],
    )

    params = {
        "video": args.video_path.name,
        "label_count": len(filler_labels),
        "detected_count": len(events),
        "transcript_word_count": len(words),
    }

    if not args.no_mlflow:
        log_mlflow_run(
            experiment_name(_DIMENSION), args.run_name, params=params, metrics=metrics
        )

    out = eval_dump_path(_DIMENSION, args.video_path.stem, args.run_name)
    write_eval_dump(
        out,
        {
            "video": args.video_path.name,
            "run_name": args.run_name,
            "params": params,
            "metrics": metrics,
            "detected": [{"start": e.start, "end": e.end, "text": e.text} for e in events],
            "labels": filler_labels,
            "label_time_tokens": _label_time_tokens(filler_labels, words),
        },
        force=args.force,
    )
    _log.info("  → dumped %s", out.name)


if __name__ == "__main__":
    main()
