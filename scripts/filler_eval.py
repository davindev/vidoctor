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

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vidoctor.audio.filler import detect_filler_events  # noqa: E402
from vidoctor.audio.transcribe import transcribe_video  # noqa: E402
from vidoctor.config import get_settings  # noqa: E402
from vidoctor.eval.labels import load_labels  # noqa: E402
from vidoctor.eval.metrics import compute_filler_metrics  # noqa: E402
from vidoctor.graph.state import Word  # noqa: E402

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


def _model_tag() -> str:
    """VIDOCTOR_WHISPER_MODEL 값에서 cache 키로 쓸 짧은 식별자 추출.

    환경변수 미설정이면 "default", 경로면 basename, HF id면 마지막 segment.
    """
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


def main() -> None:
    parser = argparse.ArgumentParser(description="filler P/R/F1 + MLflow logging")
    parser.add_argument("video_path", type=Path)
    parser.add_argument("labels_csv", type=Path)
    parser.add_argument(
        "--run-name", required=True, help="MLflow run name (예: baseline_lecture, stage1_lecture)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="transcript cache 무시하고 재추출",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true", help="MLflow 로그 생략 (디버그용)"
    )
    args = parser.parse_args()

    if not args.video_path.exists():
        sys.exit(f"video not found: {args.video_path}")
    if not args.labels_csv.exists():
        sys.exit(f"labels not found: {args.labels_csv}")

    words = _load_or_transcribe(args.video_path, args.no_cache)

    events = detect_filler_events(words)
    detected_starts = [e.start for e in events]

    labels = load_labels(args.labels_csv)
    filler_labels = [(lbl.start, lbl.end) for lbl in labels if lbl.dimension == "filler"]

    metrics = _metrics_dict(filler_labels, events)
    print(
        f"\n[{args.run_name}] filler: TP={metrics['tp']} FP={metrics['fp']} "
        f"FN={metrics['fn']} P={metrics['precision']:.3f} "
        f"R={metrics['recall']:.3f} F1={metrics['f1']:.3f}"
    )

    if not args.no_mlflow:
        settings = get_settings()
        if settings.mlflow_tracking_uri:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(_EXPERIMENT_NAME)
        with mlflow.start_run(run_name=args.run_name):
            mlflow.log_params(
                {
                    "video": args.video_path.name,
                    "label_count": len(filler_labels),
                    "detected_count": len(events),
                    "transcript_word_count": len(words),
                }
            )
            mlflow.log_metrics(metrics)
        print(f"  → mlflow run logged ({_EXPERIMENT_NAME} / {args.run_name})")

    out = ROOT / "data" / "golden" / f"filler_eval_{args.video_path.stem}_{args.run_name}.json"
    out.write_text(
        json.dumps(
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
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"  → dumped {out.name}")


if __name__ == "__main__":
    main()
