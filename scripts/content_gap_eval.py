"""content_gap 차원만 평가 — frame sampling + GPT-4o 호출 후 P/R/F1 + 비용·latency 측정.

content_gap은 LLM 1회 호출이라 baseline 측정이 곧 진단이다. 이 스크립트는:
  1. transcript 캐시 로드(없으면 추출) — filler_eval과 동일 캐시 파일 재활용
  2. content_gap.py 내부 구성요소(_sample_frames / _build_message / _RUBRICS /
     _ContentGapResponse)를 직접 호출 — production 함수에 평가 hook 주입 회피
  3. structured output을 include_raw=True로 받아 token usage·raw response를 보존
  4. 라벨 시간대에 들어간 frame 시각·transcript 텍스트·LLM 출력 reasoning 모두 dump
  5. MLflow에 P/R/F1 + LLM 호출 비용·latency·prompt_tokens 기록

사용법:
    uv run python scripts/content_gap_eval.py data/golden/lecture.mp4 \\
        data/golden/lecture_labels.csv --run-name baseline_lecture
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import cast

from vidoctor.config import ROOT
from vidoctor.eval._script_lib import (
    load_or_transcribe,
    log_mlflow_run,
)
from vidoctor.eval.labels import load_labels
from vidoctor.eval.metrics import DIM_IOU_THRESHOLD, _compute_iou_metrics
from vidoctor.graph.state import Category, ContentGapEvent, Word
from vidoctor.llm import (
    estimate_cost_usd,
    extract_token_usage,
    get_chat_model,
)
from vidoctor.vision.content_gap import (
    _RUBRICS,
    JPEG_QUALITY,
    MAX_FRAME_HEIGHT,
    MAX_SAMPLES,
    SAMPLE_INTERVAL_SEC,
    SCENE_DEDUP_THRESHOLD_SEC,
    TRANSCRIPT_WINDOW_SEC,
    _anchor_to_asr,
    _build_message,
    _ContentGapResponse,
    _sample_frames,
)

_EXPERIMENT_NAME = "vidoctor-content_gap"


async def _detect_with_meta(
    video_path: Path, transcript: list[Word], category: Category, model_name: str
) -> dict:
    """detect_content_gap_events와 동일 흐름이되 raw response·usage·latency 보존."""
    samples = _sample_frames(str(video_path), transcript)
    if not samples:
        return {
            "samples": [],
            "events": [],
            "latency_sec": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "raw_text": "",
            "issues_raw": [],
        }

    rubric = _RUBRICS[category]
    message = _build_message(samples, rubric)

    # production(detect_content_gap_events)과 동일 max_tokens 정책으로 평가.
    model = get_chat_model(model=model_name, temperature=0.0, max_tokens=1024)
    structured = model.with_structured_output(_ContentGapResponse, include_raw=True)

    t0 = time.perf_counter()
    result = await structured.ainvoke([message])
    latency = time.perf_counter() - t0

    raw = result["raw"] if isinstance(result, dict) else None
    parsed = cast(
        _ContentGapResponse,
        result["parsed"] if isinstance(result, dict) else result,
    )
    prompt_tok, completion_tok = extract_token_usage(raw)
    total_tok = prompt_tok + completion_tok
    raw_text = ""
    if raw is not None:
        content = getattr(raw, "content", "")
        raw_text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)

    events: list[ContentGapEvent] = []
    for issue in parsed.issues:
        anchored = _anchor_to_asr(issue, transcript)
        s, e = anchored if anchored is not None else (issue.start_sec, issue.end_sec)
        events.append(ContentGapEvent(start=s, end=e, description=issue.description))
    return {
        "samples": samples,
        "events": events,
        "issues_raw": [i.model_dump() for i in parsed.issues],
        "latency_sec": latency,
        "prompt_tokens": prompt_tok,
        "completion_tokens": completion_tok,
        "total_tokens": total_tok,
        "raw_text": raw_text,
    }


def _label_diagnostics(
    label_intervals: list[tuple[float, float]],
    samples: list,
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
    parser = argparse.ArgumentParser(description="content_gap P/R/F1 + cost·latency + MLflow")
    parser.add_argument("video_path", type=Path)
    parser.add_argument("labels_csv", type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--no-cache", action="store_true", help="transcript cache 무시")
    parser.add_argument("--no-mlflow", action="store_true")
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

    if not args.video_path.exists():
        sys.exit(f"video not found: {args.video_path}")
    if not args.labels_csv.exists():
        sys.exit(f"labels not found: {args.labels_csv}")

    transcript = load_or_transcribe(args.video_path, args.no_cache)

    print(f"sampling frames + invoking {args.model}...")
    category = cast(Category, args.category)
    result = asyncio.run(
        _detect_with_meta(args.video_path, transcript, category, args.model)
    )

    samples = result["samples"]
    events = result["events"]
    cost_usd = estimate_cost_usd(
        args.model, result["prompt_tokens"], result["completion_tokens"]
    )

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
        "latency_sec": round(result["latency_sec"], 3),
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "total_tokens": result["total_tokens"],
        "cost_usd": round(cost_usd, 6),
        "image_count": len(samples),
    }

    print(
        f"\n[{args.run_name}] content_gap: TP={m.tp} FP={m.fp} FN={m.fn} "
        f"P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f}"
    )
    print(
        f"  model={args.model} images={len(samples)} "
        f"latency={metrics['latency_sec']:.2f}s "
        f"prompt_tok={metrics['prompt_tokens']} "
        f"completion_tok={metrics['completion_tokens']} "
        f"cost=${metrics['cost_usd']:.4f}"
    )

    diag = _label_diagnostics(cg_intervals, samples, transcript)
    for d in diag:
        ls, le = d["label"]["start"], d["label"]["end"]
        n_frames = len(d["frames_covering_label"])
        print(
            f"  label[{ls:.0f}-{le:.0f}s]: frames_in_window={n_frames} "
            f"words_in_label={d['n_words_in_label']}"
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

    out = (
        ROOT
        / "data"
        / "golden"
        / f"content_gap_eval_{args.video_path.stem}_{args.run_name}.json"
    )
    out.write_text(
        json.dumps(
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
                "issues_raw": result["issues_raw"],
                "labels": [
                    {"start": lbl.start, "end": lbl.end, "note": lbl.note}
                    for lbl in cg_labels
                ],
                "label_diagnostics": diag,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"  → dumped {out.name}")


if __name__ == "__main__":
    main()
