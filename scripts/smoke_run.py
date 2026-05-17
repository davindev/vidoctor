"""5차원 graph 통합 스모크 — 영상 1편 흘려보내 crash·연결 점검 + 결과 dump.

사용법:
    uv run python scripts/smoke_run.py data/golden/inputs/lecture.mp4 lecture
    uv run python scripts/smoke_run.py data/golden/inputs/vlog.mp4 vlog

lecture 카테고리는 EXPECTED_LECTURE 마커와 overlap 비교 표를 추가 출력.
P/R/F1 정량 평가는 차원별 *_eval.py 사용.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from typing import TypedDict, cast

from pydantic import BaseModel

from vidoctor.config import ROOT
from vidoctor.eval._script_lib import existing_file, configure_eval_logging
from vidoctor.graph import run_analysis
from vidoctor.graph.state import (
    DIM_TO_STATE_FIELD,
    AnalysisState,
    Category,
    Dimension,
)

_log = logging.getLogger(__name__)


class ExpectedMarker(TypedDict):
    start: float
    end: float
    dimension: Dimension
    note: str


# script.md 의도된 마커. 영상 길이가 의도(3:00)와 다를 경우 매칭이 흔들리는 것은 정상.
EXPECTED_LECTURE: list[ExpectedMarker] = [
    {"start": 25, "end": 28, "dimension": "gaze", "note": "노트북 응시 3s"},
    {"start": 60, "end": 80, "dimension": "filler", "note": "음·어 burst 8개"},
    {"start": 84, "end": 90, "dimension": "gaze", "note": "노트북 응시 6s"},
    {"start": 90, "end": 122, "dimension": "dead_zone", "note": "슬라이드 5 32초 유지"},
    {"start": 95, "end": 100, "dimension": "dead_zone", "note": "5초 침묵"},
    {"start": 100, "end": 122, "dimension": "content_gap", "note": "공변성↔반공변성"},
    {"start": 122, "end": 130, "dimension": "cps", "note": "11.25 CPS too_fast"},
    {"start": 140, "end": 145, "dimension": "gaze", "note": "자료 응시 5s"},
    {"start": 155, "end": 170, "dimension": "cps", "note": "1.7 CPS too_slow"},
]

DUMP_FIELDS = (*DIM_TO_STATE_FIELD.values(), "suggestions")

# script.md 의도와 실제 영상 길이가 어긋날 때 매칭이 흔들리지 않도록 보수적 tolerance.
OVERLAP_TOL = 5.0


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float, tol: float) -> bool:
    """두 구간이 tolerance 안에서 겹치는지 검사."""
    return (a_start - tol) <= b_end and (b_start - tol) <= a_end


def _events_dump(state: AnalysisState, field: str) -> list[dict]:
    """state의 차원 필드를 JSON 직렬화 가능한 dict 리스트로 변환."""
    events = state.get(field, []) or []  # type: ignore[literal-required]
    return [
        e.model_dump(mode="json") if isinstance(e, BaseModel) else dict(e) for e in events
    ]


def _log_lecture_match_table(state: AnalysisState) -> None:
    """lecture EXPECTED_LECTURE 마커별 검출 hit 여부를 표로 로그."""
    _log.info("=== expected vs detected (lecture markers) ===")
    _log.info("%-12s %-14s %-6s note", "dim", "expected", "hits")
    _log.info("-" * 70)
    for exp in EXPECTED_LECTURE:
        events = _events_dump(state, DIM_TO_STATE_FIELD[exp["dimension"]])
        hits = [
            ev for ev in events
            if _overlaps(exp["start"], exp["end"], ev["start"], ev["end"], OVERLAP_TOL)
        ]
        rng = f"{int(exp['start'])}-{int(exp['end'])}s"
        mark = "✓" if hits else "✗"
        _log.info(
            "%-12s %-14s %s%-5d %s",
            exp["dimension"], rng, mark, len(hits), exp["note"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="5차원 graph 통합 스모크")
    parser.add_argument("video_path", type=existing_file)
    parser.add_argument("category", choices=["lecture", "vlog", "other"])
    args = parser.parse_args()

    video_path = str(args.video_path)
    category = cast(Category, args.category)
    configure_eval_logging(args.video_path.stem)

    _log.info("스모크 시작: video=%s category=%s", video_path, category)
    t0 = time.time()
    state = asyncio.run(run_analysis(video_path, category))
    elapsed = time.time() - t0
    _log.info("graph 완료: %.1fs", elapsed)

    out_path = ROOT / "data" / "golden" / "eval_dumps" / "smoke" / f"{args.video_path.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "video_path": video_path,
        "category": category,
        "elapsed_sec": round(elapsed, 1),
        "transcript_word_count": len(state.get("transcript") or []),
        **{f: _events_dump(state, f) for f in DUMP_FIELDS},
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    _log.info("dump 저장: %s", out_path)

    if category == "lecture":
        _log_lecture_match_table(state)


if __name__ == "__main__":
    main()
