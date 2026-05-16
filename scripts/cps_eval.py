"""cps 차원 단독 평가 — P/R/F1 + 윈도우 통계 + F0 multi-feature + MLflow 기록.

사용법:
    uv run python scripts/cps_eval.py data/golden/inputs/vlog.mp4 \\
        data/golden/labels/vlog_labels.csv --run-name baseline_vlog
"""

from __future__ import annotations

import logging
import statistics
from pathlib import Path

import numpy as np

from vidoctor.audio.cps import (
    F0_AND_SIGMA,
    MIN_NET_SPEECH_SEC,
    MIN_STDEV,
    MIN_WINDOWS_FOR_STATS,
    PAUSE_THRESHOLD_SEC,
    SIGMA_THRESHOLD,
    STEP_SEC,
    WINDOW_SEC,
    detect_cps_anomalies,
    sliding_windows,
)
from vidoctor.audio.pitch import (
    extract_pitch_track,
    window_pitch_features,
)
from vidoctor.config import ROOT
from vidoctor.eval._script_lib import (
    build_eval_parser,
    configure_eval_logging,
    eval_dump_path,
    experiment_name,
    filter_labels_by_dim,
    load_or_transcribe,
    log_mlflow_run,
    metrics_to_dict,
    model_tag,
    write_eval_dump,
)
from vidoctor.eval.labels import load_labels
from vidoctor.eval.metrics import compute_cps_metrics

_log = logging.getLogger(__name__)
_DIMENSION = "cps"
# 라벨 경계 ±패딩까지 진단 dump에 포함 (라벨러 컷팅 오차 흡수).
_LABEL_PADDING_SEC = 1.0


def _f0_cache_path(video_path: Path) -> Path:
    return ROOT / "data" / "golden" / "inputs" / f"f0_{video_path.stem}.npz"


def _load_or_extract_pitch(
    video_path: Path, no_cache: bool
) -> tuple[np.ndarray, np.ndarray]:
    """오디오에서 F0 시계열 추출 (npz 캐시)."""
    cache = _f0_cache_path(video_path)
    if cache.exists() and not no_cache:
        _log.info("캐시된 F0 로드: %s", cache.name)
        d = np.load(cache)
        return d["f0"], d["times"]
    _log.info("F0 추출 중: %s", video_path.name)
    f0, times = extract_pitch_track(str(video_path))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, f0=f0, times=times)
    _log.info("  → %d 프레임 (캐시 저장: %s)", len(f0), cache.name)
    return f0, times


def main() -> None:
    parser = build_eval_parser("cps P/R/F1 + MLflow logging")
    parser.add_argument(
        "--no-pitch",
        action="store_true",
        help="F0 multi-feature 비활성화",
    )
    args = parser.parse_args()
    configure_eval_logging(args.run_name)

    words = load_or_transcribe(args.video_path, args.no_cache)

    windows = sliding_windows(words)
    cps_values = [w.cps for w in windows]
    win_mean = statistics.mean(cps_values) if cps_values else 0.0
    win_std = statistics.stdev(cps_values) if len(cps_values) >= 2 else 0.0

    pitch_features = None
    if not args.no_pitch:
        f0, times = _load_or_extract_pitch(args.video_path, args.no_cache)
        pitch_features = window_pitch_features(
            f0, times, [(w.start, w.end) for w in windows]
        )

    events = detect_cps_anomalies(words, pitch_features=pitch_features)

    labels = load_labels(args.labels_csv)
    cps_labels = filter_labels_by_dim(labels, _DIMENSION)

    metrics = metrics_to_dict(compute_cps_metrics(cps_labels, events))
    _log.info(
        "cps: TP=%d FP=%d FN=%d P=%.3f R=%.3f F1=%.3f",
        metrics["tp"], metrics["fp"], metrics["fn"],
        metrics["precision"], metrics["recall"], metrics["f1"],
    )
    _log.info(
        "  windows=%d mean=%.2f std=%.2f events=%d",
        len(windows), win_mean, win_std, len(events),
    )

    params = {
        "video": args.video_path.name,
        "label_count": len(cps_labels),
        "detected_count": len(events),
        "transcript_word_count": len(words),
        "window_sec": WINDOW_SEC,
        "step_sec": STEP_SEC,
        "pause_threshold_sec": PAUSE_THRESHOLD_SEC,
        "sigma_threshold": SIGMA_THRESHOLD,
        "min_stdev": MIN_STDEV,
        "pitch_enabled": not args.no_pitch,
        "min_net_speech_sec": MIN_NET_SPEECH_SEC,
        "min_windows_for_stats": MIN_WINDOWS_FOR_STATS,
        "video_window_count": len(windows),
        "video_window_mean_cps": round(win_mean, 3),
        "video_window_std_cps": round(win_std, 3),
        "whisper_model": model_tag(),
    }
    if not args.no_pitch:
        params["f0_and_sigma"] = F0_AND_SIGMA

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
            "detected": [
                {"start": e.start, "end": e.end, "cps": e.cps, "kind": e.kind}
                for e in events
            ],
            "labels": [
                {"start": lbl.start, "end": lbl.end, "kind": lbl.kind, "note": lbl.note}
                for lbl in cps_labels
            ],
            "label_time_windows": [
                {
                    "label": {"start": lbl.start, "end": lbl.end, "kind": lbl.kind},
                    "windows_in_range": [
                        {
                            "start": w.start,
                            "end": w.end,
                            "cps": round(w.cps, 3),
                            "z": round((w.cps - win_mean) / win_std, 3)
                            if win_std > 0
                            else 0.0,
                        }
                        for w in windows
                        if w.end > lbl.start - _LABEL_PADDING_SEC
                        and w.start < lbl.end + _LABEL_PADDING_SEC
                    ],
                }
                for lbl in cps_labels
            ],
        },
        force=args.force,
    )
    _log.info("  → dumped %s", out.name)


if __name__ == "__main__":
    main()
