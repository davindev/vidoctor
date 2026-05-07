"""cps 차원만 평가 — content_gap(GPT-4o) 호출 없이 P/R/F1 측정 + MLflow 기록.

transcribe + detect_cps_anomalies만 돌려, 라벨 vs 검출 매칭 + 윈도우 통계
(mean·std·이상 비율)를 함께 dump해 σ 임계·윈도우 길이·평탄 가드 의사결정 자료를 만든다.

매칭은 `compute_cps_metrics`(라벨 ±1s 확장 IoU + kind 일치 필수)와 동일.

transcript는 영상별 JSON에 캐시되어 임계 튜닝 반복 시 transcribe 재실행 회피.
캐시 무효화는 --no-cache 옵션 또는 캐시 파일 삭제.

사용법:
    uv run python scripts/cps_eval.py data/golden/vlog.mp4 \\
        data/golden/vlog_labels.csv --run-name baseline_vlog
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from vidoctor.audio.cps import (  # noqa: E402
    F0_AND_SIGMA,
    MIN_NET_SPEECH_SEC,
    MIN_STDEV,
    MIN_WINDOWS_FOR_STATS,
    PAUSE_THRESHOLD_SEC,
    SIGMA_THRESHOLD,
    STEP_SEC,
    WINDOW_SEC,
    _sliding_windows,
    detect_cps_anomalies,
)
from vidoctor.audio.pitch import (  # noqa: E402
    extract_pitch_track,
    window_pitch_features,
)
from vidoctor.audio.transcribe import transcribe_video  # noqa: E402
from vidoctor.config import get_settings  # noqa: E402
from vidoctor.eval.labels import load_labels  # noqa: E402
from vidoctor.eval.metrics import compute_cps_metrics  # noqa: E402
from vidoctor.graph.state import Word  # noqa: E402

_EXPERIMENT_NAME = "vidoctor-cps"


def _metrics_dict(cps_labels, events) -> dict[str, float]:
    """compute_cps_metrics 결과를 mlflow.log_metrics 호환 dict로 변환."""
    m = compute_cps_metrics(cps_labels, events)
    return {
        "tp": m.tp,
        "fp": m.fp,
        "fn": m.fn,
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "temporal_iou_mean": m.temporal_iou_mean,
    }


def _model_tag() -> str:
    model = os.environ.get("VIDOCTOR_WHISPER_MODEL")
    if not model:
        return "default"
    return Path(model).name.replace("/", "_") or "default"


def _transcript_cache_path(video_path: Path) -> Path:
    return ROOT / "data" / "golden" / f"transcript_{video_path.stem}_{_model_tag()}.json"


def _load_or_transcribe(video_path: Path, no_cache: bool) -> list[Word]:
    cache = _transcript_cache_path(video_path)
    if cache.exists() and not no_cache:
        print(f"loading cached transcript: {cache.name}")
        data = json.loads(cache.read_text())
        return [Word(**w) for w in data]

    print(f"transcribing {video_path.name}...")
    words = asyncio.run(transcribe_video(str(video_path)))
    cache.write_text(
        json.dumps([w.model_dump() for w in words], ensure_ascii=False, indent=2)
    )
    print(f"  → {len(words)} words (cached → {cache.name})")
    return words


def _f0_cache_path(video_path: Path) -> Path:
    return ROOT / "data" / "golden" / f"f0_{video_path.stem}.npz"


def _load_or_extract_pitch(video_path: Path, no_cache: bool):
    """오디오에서 F0 시계열 추출 (npz cache). 동일 cache는 transcript와 무관."""
    cache = _f0_cache_path(video_path)
    if cache.exists() and not no_cache:
        print(f"loading cached F0: {cache.name}")
        d = np.load(cache)
        return d["f0"], d["times"]
    print(f"extracting F0 from {video_path.name}...")
    f0, times = extract_pitch_track(str(video_path))
    np.savez(cache, f0=f0, times=times)
    print(f"  → {len(f0)} frames (cached → {cache.name})")
    return f0, times


def main() -> None:
    parser = argparse.ArgumentParser(description="cps P/R/F1 + MLflow logging")
    parser.add_argument("video_path", type=Path)
    parser.add_argument("labels_csv", type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument(
        "--no-pitch",
        action="store_true",
        help="F0 multi-feature 비활성화 (cps z-score 단독 — 비교 측정용)",
    )
    args = parser.parse_args()

    if not args.video_path.exists():
        sys.exit(f"video not found: {args.video_path}")
    if not args.labels_csv.exists():
        sys.exit(f"labels not found: {args.labels_csv}")

    words = _load_or_transcribe(args.video_path, args.no_cache)

    windows = _sliding_windows(words)
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
    cps_labels = [lbl for lbl in labels if lbl.dimension == "cps"]

    metrics = _metrics_dict(cps_labels, events)
    print(
        f"\n[{args.run_name}] cps: TP={metrics['tp']} FP={metrics['fp']} "
        f"FN={metrics['fn']} P={metrics['precision']:.3f} "
        f"R={metrics['recall']:.3f} F1={metrics['f1']:.3f}"
    )
    print(
        f"  windows={len(windows)} mean={win_mean:.2f} std={win_std:.2f} "
        f"events={len(events)}"
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
        "whisper_model": _model_tag(),
    }
    if not args.no_pitch:
        params["f0_and_sigma"] = F0_AND_SIGMA

    if not args.no_mlflow:
        settings = get_settings()
        if settings.mlflow_tracking_uri:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(_EXPERIMENT_NAME)
        with mlflow.start_run(run_name=args.run_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
        print(f"  → mlflow run logged ({_EXPERIMENT_NAME} / {args.run_name})")

    out = ROOT / "data" / "golden" / f"cps_eval_{args.video_path.stem}_{args.run_name}.json"
    out.write_text(
        json.dumps(
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
                            if w.end > lbl.start - 1.0 and w.start < lbl.end + 1.0
                        ],
                    }
                    for lbl in cps_labels
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"  → dumped {out.name}")


if __name__ == "__main__":
    main()
