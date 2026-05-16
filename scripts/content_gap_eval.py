"""content_gap 차원만 평가 — frame sampling + GPT-4o 호출 후 P/R/F1 + 비용·latency 측정.

content_gap은 LLM 1회 호출이라 baseline 측정이 곧 진단이다. 이 스크립트는:
  1. transcript 캐시 로드(없으면 추출) — filler_eval과 동일 캐시 파일 재활용
  2. content_gap.detect_with_diagnostics를 호출 — production 흐름과 동일한 진입점,
     raw response·token usage·latency까지 함께 받음
  3. 라벨 시간대에 들어간 frame 시각·transcript 텍스트·LLM 출력 reasoning 모두 dump
  4. MLflow에 P/R/F1 + LLM 호출 비용·latency·prompt_tokens 기록

사용법:
    uv run python scripts/content_gap_eval.py data/golden/lecture.mp4 \\
        data/golden/lecture_labels.csv --run-name baseline_lecture
"""

from __future__ import annotations

import asyncio
import logging
from typing import cast

from vidoctor.eval._script_lib import (
    build_eval_parser,
    configure_eval_logging,
    eval_dump_path,
    load_or_transcribe,
    log_mlflow_run,
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
_EXPERIMENT_NAME = "vidoctor-content_gap"


def _label_diagnostics(
    label_intervals: list[tuple[float, float]],
    samples: list[FrameSample],
    transcript: list[Word],
) -> list[dict]:
    """라벨 시간대에 들어간 frame·transcript 진단.

    라벨 [ls, le]의 ±TRANSCRIPT_WINDOW_SEC 안에 frame이 한 장이라도 있는지가 결정적.
    없으면 LLM이 라벨 구간을 input으로 못 보고 그 라벨은 구조적 FN.
    """
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
        help="LLM model id (gpt-4o / gpt-4o-mini 등)",
    )
    args = parser.parse_args()
    configure_eval_logging(args.run_name)

    transcript = load_or_transcribe(args.video_path, args.no_cache)

    _log.info("sampling frames + invoking %s...", args.model)
    category = cast(Category, args.category)
    diag = asyncio.run(
        detect_with_diagnostics(
            str(args.video_path), transcript, category, model_name=args.model
        )
    )

    samples = diag.samples
    events = diag.events

    labels = load_labels(args.labels_csv)
    cg_labels = [lbl for lbl in labels if lbl.dimension == "content_gap"]
    cg_intervals = [(lbl.start, lbl.end) for lbl in cg_labels]

    m = _compute_iou_metrics("content_gap", cg_intervals, events)
    metrics = {
        "tp": m.tp,
        "fp": m.fp,
        "fn": m.fn,
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "temporal_iou_mean": m.temporal_iou_mean,
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

    label_diag = _label_diagnostics(cg_intervals, samples, transcript)
    for d in label_diag:
        ls, le = d["label"]["start"], d["label"]["end"]
        n_frames = len(d["frames_covering_label"])
        _log.info(
            "  label[%.0f-%.0fs]: frames_in_window=%d words_in_label=%d",
            ls, le, n_frames, d["n_words_in_label"],
        )

    params = {
        "video": args.video_path.name,
        "category": args.category,
        "model": args.model,
        "label_count": len(cg_labels),
        "detected_count": len(events),
        "sample_interval_sec": SAMPLE_INTERVAL_SEC,
        "transcript_window_sec": TRANSCRIPT_WINDOW_SEC,
        "max_samples": MAX_SAMPLES,
        "max_frame_height": MAX_FRAME_HEIGHT,
        "jpeg_quality": JPEG_QUALITY,
        "scene_dedup_threshold_sec": SCENE_DEDUP_THRESHOLD_SEC,
        "iou_threshold": DIM_IOU_THRESHOLD["content_gap"],
        "image_count": len(samples),
    }

    if not args.no_mlflow:
        log_mlflow_run(_EXPERIMENT_NAME, args.run_name, params=params, metrics=metrics)

    out = eval_dump_path("content_gap", args.video_path.stem, args.run_name)
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
                for lbl in cg_labels
            ],
            "label_diagnostics": label_diag,
        },
        force=args.force,
    )
    _log.info("  → dumped %s", out.name)


if __name__ == "__main__":
    main()
