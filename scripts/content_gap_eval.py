"""content_gap 차원 단독 평가 — GPT-4o Vision 호출 + P/R/F1 + 비용·latency + MLflow.

사용법:
    uv run python scripts/content_gap_eval.py data/golden/inputs/lecture.mp4 \\
        data/golden/labels/lecture_labels.csv --run-name baseline_lecture
"""

from __future__ import annotations

import asyncio
import logging
from typing import cast

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
from vidoctor.eval.metrics import DIM_IOU_THRESHOLD, _compute_iou_metrics
from vidoctor.graph.state import Category, Word
from vidoctor.vision.content_gap import (
    JPEG_QUALITY,
    MAX_FRAME_HEIGHT,
    MAX_SAMPLES,
    SAMPLE_INTERVAL_SEC,
    SCENE_DEDUP_THRESHOLD_SEC,
    TRANSCRIPT_WINDOW_SEC,
    FrameSample,
    detect_with_diagnostics,
)

_log = logging.getLogger(__name__)
_DIMENSION = "content_gap"


def _label_diagnostics(
    label_intervals: list[tuple[float, float]],
    samples: list[FrameSample],
    transcript: list[Word],
) -> list[dict]:
    """라벨 시간대별 frame 커버리지 + transcript 진단."""
    out: list[dict] = []
    for ls, le in label_intervals:
        center = (ls + le) / 2
        in_window = [
            {"time_sec": round(s.time_sec, 2), "transcript_len": len(s.transcript_text)}
            for s in samples
            if s.time_sec >= ls - TRANSCRIPT_WINDOW_SEC
            and s.time_sec <= le + TRANSCRIPT_WINDOW_SEC
        ]
        words_in = [w for w in transcript if ls <= w.start <= le]
        out.append(
            {
                "label": {"start": ls, "end": le, "center": center},
                "frames_covering_label": in_window,
                "transcript_in_label": " ".join(w.text for w in words_in)[:400],
                "n_words_in_label": len(words_in),
            }
        )
    return out


def main() -> None:
    parser = build_eval_parser("content_gap P/R/F1 + cost·latency + MLflow")
    parser.add_argument(
        "--category",
        default="lecture",
        choices=["lecture", "other"],
        help="rubric 선택 (vlog는 비활성)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="LLM 모델 이름 (예: gpt-4o, gpt-4o-mini)",
    )
    args = parser.parse_args()
    configure_eval_logging(args.run_name)

    transcript = load_or_transcribe(args.video_path, args.no_cache)

    _log.info("프레임 샘플링 + LLM 호출 (%s)", args.model)
    category = cast(Category, args.category)
    diag = asyncio.run(
        detect_with_diagnostics(
            str(args.video_path), transcript, category, model_name=args.model
        )
    )

    samples = diag.samples
    events = diag.events

    labels = load_labels(args.labels_csv)
    content_gap_labels = filter_labels_by_dim(labels, _DIMENSION)
    content_gap_intervals = [(lbl.start, lbl.end) for lbl in content_gap_labels]

    m = _compute_iou_metrics(_DIMENSION, content_gap_intervals, events)
    metrics = metrics_to_dict(m) | {
        "latency_sec": round(diag.latency_sec, 3),
        "prompt_tokens": diag.prompt_tokens,
        "completion_tokens": diag.completion_tokens,
        "total_tokens": diag.total_tokens,
        "cost_usd": round(diag.cost_usd, 6),
        "image_count": len(samples),
    }

    _log.info(
        "content_gap: TP=%d FP=%d FN=%d P=%.3f R=%.3f F1=%.3f",
        m.tp, m.fp, m.fn, m.precision, m.recall, m.f1,
    )
    _log.info(
        "  model=%s images=%d latency=%.2fs "
        "prompt_tok=%d completion_tok=%d cost=$%.4f",
        args.model, len(samples), metrics["latency_sec"],
        metrics["prompt_tokens"], metrics["completion_tokens"], metrics["cost_usd"],
    )

    label_diag = _label_diagnostics(content_gap_intervals, samples, transcript)
    for d in label_diag:
        ls, le = d["label"]["start"], d["label"]["end"]
        n_frames = len(d["frames_covering_label"])
        _log.info(
            "  label[%.0f-%.0fs]: frames_in_window=%d words_in_label=%d",
            ls, le, n_frames, d["n_words_in_label"],
        )

    params = {
        "video": args.video_path.name,
        "label_count": len(content_gap_labels),
        "detected_count": len(events),
        "category": category,
        "model": args.model,
        "sample_interval_sec": SAMPLE_INTERVAL_SEC,
        "transcript_window_sec": TRANSCRIPT_WINDOW_SEC,
        "max_samples": MAX_SAMPLES,
        "max_frame_height": MAX_FRAME_HEIGHT,
        "jpeg_quality": JPEG_QUALITY,
        "scene_dedup_threshold_sec": SCENE_DEDUP_THRESHOLD_SEC,
        "iou_threshold": DIM_IOU_THRESHOLD[_DIMENSION],
        "image_count": len(samples),
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
            "frame_samples": [
                {
                    "time_sec": round(s.time_sec, 2),
                    "transcript_text": s.transcript_text,
                }
                for s in samples
            ],
            "detected": [
                {"start": e.start, "end": e.end, "description": e.description}
                for e in events
            ],
            "issues_raw": diag.issues_raw,
            "labels": [
                {"start": lbl.start, "end": lbl.end, "note": lbl.note}
                for lbl in content_gap_labels
            ],
            "label_diagnostics": label_diag,
        },
        force=args.force,
    )
    _log.info("  → dumped %s", out.name)


if __name__ == "__main__":
    main()
