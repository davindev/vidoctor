"""dead_zone 차원 단독 평가 — Silero VAD + Farneback flow 게이트 + P/R/F1 + MLflow 기록.

사용법:
    uv run python scripts/dead_zone_eval.py data/golden/inputs/lecture.mp4 \\
        data/golden/labels/lecture_labels.csv lecture --run-name baseline_lecture
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from vidoctor.config import ROOT
from vidoctor.eval._script_lib import (
    build_eval_parser,
    configure_eval_logging,
    eval_dump_path,
    experiment_name,
    filter_labels_by_dim,
    log_mlflow_run,
    metrics_to_dict,
    write_eval_dump,
)
from vidoctor.eval.labels import load_labels
from vidoctor.eval.metrics import DIM_IOU_THRESHOLD, _compute_iou_metrics
from vidoctor.graph.state import Category
from vidoctor.vision.dead_zone import (
    CATEGORY_CONFIG,
    DOWNSAMPLE_HEIGHT,
    FRAME_SAMPLE_FPS,
    VAD_MIN_SILENCE_MS,
    DeadZoneEvent,
    SilentInterval,
    flow_median_in,
    flow_series,
    load_audio_or_empty,
    silent_intervals_from_audio,
)

_log = logging.getLogger(__name__)
_DIMENSION = "dead_zone"


def _flow_cache_path(video_path: Path) -> Path:
    # 캐시 키에 통계 종류('max') 명시 — 통계 정의가 바뀌면 별도 캐시로 자동 분리.
    return (
        ROOT
        / "data"
        / "golden"
        / "inputs"
        / f"flow_max_{video_path.stem}_{int(FRAME_SAMPLE_FPS)}fps_{DOWNSAMPLE_HEIGHT}p.npz"
    )


def _load_or_extract_flow(
    video_path: Path, no_cache: bool
) -> tuple[np.ndarray, np.ndarray, float]:
    cache = _flow_cache_path(video_path)
    if cache.exists() and not no_cache:
        _log.info("loading cached flow: %s", cache.name)
        d = np.load(cache)
        return d["curr_times"], d["flows"], float(d["duration"])

    _log.info("computing optical flow for %s...", video_path.name)
    curr_t, flows, duration = flow_series(str(video_path))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, curr_times=curr_t, flows=flows, duration=duration)
    _log.info("  → %d samples, duration=%.1fs (cached → %s)", len(flows), duration, cache.name)
    return curr_t, flows, duration


def _detect(
    silent: list[SilentInterval],
    curr_times: np.ndarray,
    flows: np.ndarray,
    min_duration: float,
    flow_threshold: float,
) -> list[DeadZoneEvent]:
    events: list[DeadZoneEvent] = []
    for iv in silent:
        if iv.end - iv.start < min_duration:
            continue
        median = flow_median_in(curr_times, flows, iv.start, iv.end)
        if median is None or median > flow_threshold:
            continue
        events.append(DeadZoneEvent(start=iv.start, end=iv.end))
    return events


def _label_diagnostics(
    label_intervals: list[tuple[float, float]],
    silent: list[SilentInterval],
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
    configure_eval_logging(args.run_name)

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
    _log.info("loading audio + VAD...")
    audio = load_audio_or_empty(str(args.video_path))
    silent = silent_intervals_from_audio(audio, duration)
    _log.info("  → %d silent intervals", len(silent))

    events = _detect(silent, curr_times, flows, min_duration, flow_threshold)

    labels = load_labels(args.labels_csv)
    dz_labels = filter_labels_by_dim(labels, _DIMENSION)
    dz_intervals = [(lbl.start, lbl.end) for lbl in dz_labels]

    m = _compute_iou_metrics("dead_zone", dz_intervals, events)
    metrics = metrics_to_dict(m)
    _log.info(
        "dead_zone(%s): TP=%d FP=%d FN=%d P=%.3f R=%.3f F1=%.3f",
        category, m.tp, m.fp, m.fn, m.precision, m.recall, m.f1,
    )
    _log.info(
        "  min_dur=%.1fs flow_max=%.3f silent_intervals=%d events=%d",
        min_duration, flow_threshold, len(silent), len(events),
    )

    diag = _label_diagnostics(dz_intervals, silent, curr_times, flows)

    params = {
        "video": args.video_path.name,
        "label_count": len(dz_labels),
        "detected_count": len(events),
        "category": category,
        "video_duration": round(duration, 2),
        "min_duration_sec": min_duration,
        "flow_threshold": flow_threshold,
        "vad_min_silence_ms": VAD_MIN_SILENCE_MS,
        "frame_sample_fps": FRAME_SAMPLE_FPS,
        "downsample_height": DOWNSAMPLE_HEIGHT,
        "iou_threshold": DIM_IOU_THRESHOLD[_DIMENSION],
        "silent_interval_count": len(silent),
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
        force=args.force,
    )
    _log.info("  → dumped %s", out.name)


if __name__ == "__main__":
    main()
