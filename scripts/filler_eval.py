"""filler 차원만 평가 — content_gap(GPT-4o) 호출 없이 P/R/F1 측정 + MLflow 기록.

전체 graph가 아닌 transcribe + detect_filler만 돌려, 라벨 vs 검출 매칭 + 라벨
시간대 ASR 토큰을 함께 dump해 사전·임계 튜닝 의사결정 자료를 만든다.

transcript는 영상별 JSON에 캐시되어 사전 튜닝 반복 시 transcribe 재실행 회피.
캐시 무효화는 --no-cache 옵션 또는 캐시 파일 삭제.

사용법:
    uv run python scripts/filler_eval.py data/golden/lecture.mp4 \\
        data/golden/lecture_labels.csv --run-name baseline_lecture
"""

from __future__ import annotations

import logging

from vidoctor.audio.filler import detect_filler_events
from vidoctor.eval._script_lib import (
    build_eval_parser,
    configure_eval_logging,
    eval_dump_path,
    load_or_transcribe,
    log_mlflow_run,
    write_eval_dump,
)
from vidoctor.eval.labels import load_labels
from vidoctor.eval.metrics import compute_filler_metrics

_log = logging.getLogger(__name__)
_EXPERIMENT_NAME = "vidoctor-filler"


def _metrics_dict(label_intervals, events) -> dict[str, float]:
    """compute_filler_metrics 결과를 mlflow.log_metrics 호환 dict로 변환."""
    m = compute_filler_metrics(label_intervals, events)
    return {
        "tp": m.tp,
        "fp": m.fp,
        "fn": m.fn,
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
    }


def main() -> None:
    parser = build_eval_parser("filler P/R/F1 + MLflow logging")
    args = parser.parse_args()
    configure_eval_logging(args.run_name)

    words = load_or_transcribe(args.video_path, args.no_cache)

    events = detect_filler_events(words)

    labels = load_labels(args.labels_csv)
    filler_labels = [(lbl.start, lbl.end) for lbl in labels if lbl.dimension == "filler"]

    metrics = _metrics_dict(filler_labels, events)
    _log.info(
        "filler: TP=%d FP=%d FN=%d P=%.3f R=%.3f F1=%.3f",
        metrics["tp"], metrics["fp"], metrics["fn"],
        metrics["precision"], metrics["recall"], metrics["f1"],
    )

    if not args.no_mlflow:
        log_mlflow_run(
            _EXPERIMENT_NAME,
            args.run_name,
            params={
                "video": args.video_path.name,
                "label_count": len(filler_labels),
                "detected_count": len(events),
                "transcript_word_count": len(words),
            },
            metrics=metrics,
        )

    out = eval_dump_path("filler", args.video_path.stem, args.run_name)
    write_eval_dump(
        out,
        {
            "video": args.video_path.name,
            "run_name": args.run_name,
            "metrics": metrics,
            "detected": [{"start": e.start, "end": e.end, "text": e.text} for e in events],
            "labels": filler_labels,
            "label_time_tokens": [
                {
                    "label": [ls, le],
                    "tokens": [
                        {"start": w.start, "end": w.end, "text": w.text}
                        for w in words
                        if ls <= w.start < le
                    ],
                }
                for ls, le in filler_labels
            ],
        },
        force=args.force,
    )
    _log.info("  → dumped %s", out.name)


if __name__ == "__main__":
    main()
