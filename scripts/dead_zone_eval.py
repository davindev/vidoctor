"""dead_zone 차원만 평가 — content_gap(GPT-4o) 호출 없이 P/R/F1 측정 + MLflow 기록.

VAD(Silero) + SSIM 시계열 + 시간 가드로 검출. 라벨 vs 검출 매칭 + 라벨 시간대 SSIM 누적
정적 시간·VAD silent coverage 진단 dump해 임계·결합 정책 의사결정 자료를 만든다.

매칭은 IoU greedy 1:1 (IoU 임계 0.3).

SSIM 시계열은 영상·다운스케일·fps 동일하면 결정적이라 npz 캐시.
캐시 무효화는 --no-cache 옵션 또는 캐시 파일 삭제.

사용법:
    uv run python scripts/dead_zone_eval.py data/golden/lecture.mp4 \\
        data/golden/lecture_labels.csv lecture --run-name baseline_lecture
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from vidoctor.config import ROOT
from vidoctor.eval._script_lib import build_eval_parser, log_mlflow_run
from vidoctor.eval.labels import load_labels
from vidoctor.eval.metrics import DIM_IOU_THRESHOLD, _compute_iou_metrics
from vidoctor.graph.state import Category
from vidoctor.vision.dead_zone import (
    CATEGORY_CONFIG,
    DOWNSAMPLE_HEIGHT,
    FRAME_SAMPLE_FPS,
    VAD_MIN_SILENCE_MS,
    DeadZoneEvent,
    _flow_median_in,
    _flow_series,
    _Interval,
    _load_audio_or_empty,
    _silent_intervals_from_audio,
)

_EXPERIMENT_NAME = "vidoctor-dead_zone"


def _flow_cache_path(video_path: Path) -> Path:
    # 캐시 키에 통계 종류('max') 명시 — 통계 정의가 바뀌면 별도 캐시로 자동 분리.
    return (
        ROOT
        / "data"
        / "golden"
        / f"flow_max_{video_path.stem}_{int(FRAME_SAMPLE_FPS)}fps_{DOWNSAMPLE_HEIGHT}p.npz"
    )


def _load_or_extract_flow(
    video_path: Path, no_cache: bool
) -> tuple[np.ndarray, np.ndarray, float]:
    cache = _flow_cache_path(video_path)
    if cache.exists() and not no_cache:
        print(f"loading cached flow: {cache.name}")
        d = np.load(cache)
        return d["curr_times"], d["flows"], float(d["duration"])

    print(f"computing optical flow for {video_path.name}...")
    curr_t, flows, duration = _flow_series(str(video_path))
    np.savez(cache, curr_times=curr_t, flows=flows, duration=duration)
    print(f"  → {len(flows)} samples, duration={duration:.1f}s (cached → {cache.name})")
    return curr_t, flows, duration


def _detect(
    silent: list[_Interval],
    curr_times: np.ndarray,
    flows: np.ndarray,
    min_duration: float,
    flow_threshold: float,
) -> list[DeadZoneEvent]:
    events: list[DeadZoneEvent] = []
    for iv in silent:
        if iv.end - iv.start < min_duration:
            continue
        median = _flow_median_in(curr_times, flows, iv.start, iv.end)
        if median is None or median > flow_threshold:
            continue
        events.append(DeadZoneEvent(start=iv.start, end=iv.end))
    return events


def _label_diagnostics(
    label_intervals: list[tuple[float, float]],
    silent: list[_Interval],
    curr_times: np.ndarray,
    flows: np.ndarray,
) -> list[dict]:
    out: list[dict] = []
    for ls, le in label_intervals:
        ldur = max(le - ls, 1e-9)
        silent_cov = (
            sum(max(0.0, min(le, iv.end) - max(ls, iv.start)) for iv in silent) / ldur
        )
        mask = (curr_times >= ls) & (curr_times <= le)
        f_in = flows[mask]
        flow_stats = (
            {
                "n": int(f_in.size),
                "min": float(f_in.min()),
                "median": float(np.median(f_in)),
                "max": float(f_in.max()),
            }
            if f_in.size > 0
            else {"n": 0}
        )
        out.append(
            {
                "label": {"start": ls, "end": le, "duration": le - ls},
                "silent_coverage": round(silent_cov, 3),
                "flow_median_in_label": flow_stats.get("median"),
                "flow_in_label": flow_stats,
            }
        )
    return out


def main() -> None:
    parser = build_eval_parser("dead_zone P/R/F1 + MLflow logging (VAD)")
    parser.add_argument("category", choices=["lecture", "vlog", "other"])
    parser.add_argument("--min-duration", type=float, default=None)
    parser.add_argument(
        "--flow-threshold",
        type=float,
        default=None,
        help="후보 안 per-frame max 시계열의 median이 이 값 이하이면 정적. "
        "기본=카테고리 상수 (lecture 0.5, vlog/other 5.0).",
    )
    args = parser.parse_args()

    if not args.video_path.exists():
        sys.exit(f"video not found: {args.video_path}")
    if not args.labels_csv.exists():
        sys.exit(f"labels not found: {args.labels_csv}")

    category: Category = args.category
    cfg = CATEGORY_CONFIG[category]
    min_duration = (
        args.min_duration if args.min_duration is not None else cfg.min_duration_sec
    )
    flow_threshold = (
        args.flow_threshold
        if args.flow_threshold is not None
        else cfg.flow_max_threshold
    )

    curr_times, flows, duration = _load_or_extract_flow(args.video_path, args.no_cache)
    print("loading audio + VAD...")
    audio = _load_audio_or_empty(str(args.video_path))
    silent = _silent_intervals_from_audio(audio, duration)
    print(f"  → {len(silent)} silent intervals")

    events = _detect(silent, curr_times, flows, min_duration, flow_threshold)

    labels = load_labels(args.labels_csv)
    dz_labels = [lbl for lbl in labels if lbl.dimension == "dead_zone"]
    dz_intervals = [(lbl.start, lbl.end) for lbl in dz_labels]

    m = _compute_iou_metrics("dead_zone", dz_intervals, events)
    metrics = {
        "tp": m.tp,
        "fp": m.fp,
        "fn": m.fn,
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "temporal_iou_mean": m.temporal_iou_mean,
    }
    print(
        f"\n[{args.run_name}] dead_zone({category}): TP={m.tp} FP={m.fp} FN={m.fn} "
        f"P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f}"
    )
    print(
        f"  min_dur={min_duration:.1f}s flow_max={flow_threshold:.3f}\n"
        f"  silent_intervals={len(silent)} events={len(events)}"
    )

    diag = _label_diagnostics(dz_intervals, silent, curr_times, flows)

    params = {
        "video": args.video_path.name,
        "category": category,
        "label_count": len(dz_labels),
        "detected_count": len(events),
        "video_duration": round(duration, 2),
        "min_duration_sec": min_duration,
        "flow_threshold": flow_threshold,
        "vad_min_silence_ms": VAD_MIN_SILENCE_MS,
        "frame_sample_fps": FRAME_SAMPLE_FPS,
        "downsample_height": DOWNSAMPLE_HEIGHT,
        "iou_threshold": DIM_IOU_THRESHOLD["dead_zone"],
        "silent_interval_count": len(silent),
    }

    if not args.no_mlflow:
        log_mlflow_run(_EXPERIMENT_NAME, args.run_name, params=params, metrics=metrics)

    out = (
        ROOT
        / "data"
        / "golden"
        / f"dead_zone_eval_{args.video_path.stem}_{args.run_name}.json"
    )
    out.write_text(
        json.dumps(
            {
                "video": args.video_path.name,
                "category": category,
                "run_name": args.run_name,
                "params": params,
                "metrics": metrics,
                "detected": [{"start": e.start, "end": e.end} for e in events],
                "labels": [
                    {"start": lbl.start, "end": lbl.end, "note": lbl.note}
                    for lbl in dz_labels
                ],
                "label_diagnostics": diag,
                "silent_intervals": [{"start": iv.start, "end": iv.end} for iv in silent],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"  → dumped {out.name}")


if __name__ == "__main__":
    main()
