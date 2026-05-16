"""gaze 차원 단독 평가 — MediaPipe head pose + baseline 차감 + P/R/F1 + MLflow 기록.

사용법:
    uv run python scripts/gaze_eval.py data/golden/inputs/lecture.mp4 \\
        data/golden/labels/lecture_labels.csv --run-name baseline_lecture
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
from vidoctor.vision.gaze import (
    MERGE_GAP_SEC,
    MIN_DURATION_SEC,
    PITCH_THRESHOLD_DEG,
    SAMPLE_FPS,
    YAW_THRESHOLD_DEG,
    GazeConfig,
    PoseSample,
    sample_video_pose,
    samples_to_events,
    subtract_baseline,
)

_log = logging.getLogger(__name__)
_DIMENSION = "gaze"


def _pose_cache_path(video_path: Path) -> Path:
    return (
        ROOT
        / "data"
        / "golden"
        / "inputs"
        / f"gaze_pose_{video_path.stem}_{int(SAMPLE_FPS)}fps.npz"
    )


def _load_or_extract_pose(
    video_path: Path, no_cache: bool
) -> list[PoseSample]:
    cache = _pose_cache_path(video_path)
    if cache.exists() and not no_cache:
        _log.info("loading cached pose: %s", cache.name)
        d = np.load(cache)
        ts, yaws, pitches = d["t"], d["yaw"], d["pitch"]
        return [
            PoseSample(t=float(ts[i]), yaw=float(yaws[i]), pitch=float(pitches[i]))
            for i in range(len(ts))
        ]

    _log.info("extracting head pose for %s (ROI + landmarker, slow)...", video_path.name)
    samples = sample_video_pose(str(video_path))
    if not samples:
        _log.warning("  → 0 samples (ROI 추정 실패 — 화자 얼굴 미검출)")
        return samples
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        t=np.array([s.t for s in samples], dtype=np.float64),
        yaw=np.array([s.yaw for s in samples], dtype=np.float64),
        pitch=np.array([s.pitch for s in samples], dtype=np.float64),
    )
    _log.info("  → %d samples (cached → %s)", len(samples), cache.name)
    return samples


def _detect_with_thresholds(
    samples: list[PoseSample],
    yaw_thr: float,
    pitch_thr: float,
    min_duration: float,
    merge_gap: float,
) -> list:
    """sweep 임계로 samples_to_events 호출 — GazeConfig 키워드로 주입."""
    cfg = GazeConfig(
        yaw_threshold_deg=yaw_thr,
        pitch_threshold_deg=pitch_thr,
        min_duration_sec=min_duration,
        merge_gap_sec=merge_gap,
    )
    return samples_to_events(samples, cfg)


def _label_diagnostics(
    label_intervals: list[tuple[float, float]],
    samples: list[PoseSample],
    yaw_thr: float,
    pitch_thr: float,
) -> list[dict]:
    """라벨 시간대 yaw/pitch 통계 + 임계 초과 비율 진단.

    라벨 구간 안 sample이 0이면 detection 실패(landmark 못 잡음)로 간주 — gaze 검출
    파이프라인의 프레임 누락 위치를 진단할 수 있다.
    """
    if not samples:
        return [
            {
                "label": {"start": ls, "end": le, "duration": le - ls},
                "n_samples": 0,
                "note": "no samples (ROI 추정 실패 또는 landmark 미검출)",
            }
            for ls, le in label_intervals
        ]

    ts = np.array([s.t for s in samples])
    yaws = np.array([s.yaw for s in samples])
    pitches = np.array([s.pitch for s in samples])
    out: list[dict] = []
    for ls, le in label_intervals:
        mask = (ts >= ls) & (ts <= le)
        y_in, p_in = yaws[mask], pitches[mask]
        n = int(mask.sum())
        if n == 0:
            out.append(
                {
                    "label": {"start": ls, "end": le, "duration": le - ls},
                    "n_samples": 0,
                    "note": "구간 내 sample 0 — landmark 검출 실패",
                }
            )
            continue
        off_yaw = float(np.mean(np.abs(y_in) > yaw_thr))
        off_pitch = float(np.mean(np.abs(p_in) > pitch_thr))
        off_any = float(np.mean((np.abs(y_in) > yaw_thr) | (np.abs(p_in) > pitch_thr)))
        out.append(
            {
                "label": {"start": ls, "end": le, "duration": le - ls},
                "n_samples": n,
                "yaw": {
                    "min": float(y_in.min()),
                    "median": float(np.median(y_in)),
                    "max": float(y_in.max()),
                    "abs_max": float(np.max(np.abs(y_in))),
                },
                "pitch": {
                    "min": float(p_in.min()),
                    "median": float(np.median(p_in)),
                    "max": float(p_in.max()),
                    "abs_max": float(np.max(np.abs(p_in))),
                },
                "off_ratio": {
                    "yaw_only": round(off_yaw, 3),
                    "pitch_only": round(off_pitch, 3),
                    "any": round(off_any, 3),
                },
            }
        )
    return out


def _global_pose_summary(samples: list[PoseSample]) -> dict:
    if not samples:
        return {"n_samples": 0}
    yaws = np.array([s.yaw for s in samples])
    pitches = np.array([s.pitch for s in samples])
    return {
        "n_samples": len(samples),
        "duration_covered": round(float(samples[-1].t - samples[0].t), 2),
        "yaw": {
            "median": float(np.median(yaws)),
            "p10": float(np.percentile(yaws, 10)),
            "p90": float(np.percentile(yaws, 90)),
            "abs_p95": float(np.percentile(np.abs(yaws), 95)),
        },
        "pitch": {
            "median": float(np.median(pitches)),
            "p10": float(np.percentile(pitches, 10)),
            "p90": float(np.percentile(pitches, 90)),
            "abs_p95": float(np.percentile(np.abs(pitches), 95)),
        },
    }


def main() -> None:
    parser = build_eval_parser("gaze P/R/F1 + MLflow logging")
    parser.add_argument(
        "--yaw-threshold",
        type=float,
        default=None,
        help=f"yaw 임계 (deg). 기본={YAW_THRESHOLD_DEG}",
    )
    parser.add_argument(
        "--pitch-threshold",
        type=float,
        default=None,
        help=f"pitch 임계 (deg). 기본={PITCH_THRESHOLD_DEG}",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=None,
        help=f"최소 지속 시간 (s). 기본={MIN_DURATION_SEC}",
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=None,
        help=f"인접 이벤트 병합 갭 (s). 기본={MERGE_GAP_SEC}",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="baseline 차감을 끄고 절대 yaw/pitch에 임계 적용 (디버그용).",
    )
    args = parser.parse_args()
    configure_eval_logging(args.run_name)

    yaw_thr = args.yaw_threshold if args.yaw_threshold is not None else YAW_THRESHOLD_DEG
    pitch_thr = (
        args.pitch_threshold if args.pitch_threshold is not None else PITCH_THRESHOLD_DEG
    )
    min_dur = args.min_duration if args.min_duration is not None else MIN_DURATION_SEC
    merge_gap = args.merge_gap if args.merge_gap is not None else MERGE_GAP_SEC

    samples = _load_or_extract_pose(args.video_path, args.no_cache)

    baseline_yaw = baseline_pitch = 0.0
    if not args.no_baseline:
        samples, baseline_yaw, baseline_pitch = subtract_baseline(samples)
        _log.info(
            "  → baseline subtracted: yaw_median=%+.2f° pitch_median=%+.2f°",
            baseline_yaw, baseline_pitch,
        )

    events = _detect_with_thresholds(samples, yaw_thr, pitch_thr, min_dur, merge_gap)

    labels = load_labels(args.labels_csv)
    gaze_labels = filter_labels_by_dim(labels, _DIMENSION)
    gaze_intervals = [(lbl.start, lbl.end) for lbl in gaze_labels]

    m = _compute_iou_metrics("gaze", gaze_intervals, events)
    metrics = metrics_to_dict(m)
    _log.info(
        "gaze: TP=%d FP=%d FN=%d P=%.3f R=%.3f F1=%.3f",
        m.tp, m.fp, m.fn, m.precision, m.recall, m.f1,
    )
    _log.info(
        "  yaw_thr=%.1f° pitch_thr=%.1f° min_dur=%.1fs merge_gap=%.2fs "
        "pose_samples=%d detected_events=%d",
        yaw_thr, pitch_thr, min_dur, merge_gap, len(samples), len(events),
    )

    diag = _label_diagnostics(gaze_intervals, samples, yaw_thr, pitch_thr)
    pose_summary = _global_pose_summary(samples)

    params = {
        "video": args.video_path.name,
        "label_count": len(gaze_labels),
        "detected_count": len(events),
        "yaw_threshold_deg": yaw_thr,
        "pitch_threshold_deg": pitch_thr,
        "min_duration_sec": min_dur,
        "merge_gap_sec": merge_gap,
        "sample_fps": SAMPLE_FPS,
        "iou_threshold": DIM_IOU_THRESHOLD[_DIMENSION],
        "pose_sample_count": len(samples),
        "baseline_subtracted": not args.no_baseline,
        "baseline_yaw": round(baseline_yaw, 3),
        "baseline_pitch": round(baseline_pitch, 3),
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
            "pose_summary": pose_summary,
            "detected": [
                {"start": e.start, "end": e.end, "direction": e.direction}
                for e in events
            ],
            "labels": [
                {"start": lbl.start, "end": lbl.end, "note": lbl.note}
                for lbl in gaze_labels
            ],
            "label_diagnostics": diag,
        },
        force=args.force,
    )
    _log.info("  → dumped %s", out.name)


if __name__ == "__main__":
    main()
