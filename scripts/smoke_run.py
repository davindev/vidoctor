"""골든셋 영상 1편을 5차원 graph에 흘려보내 결과를 덤프한다.

사용법:
    uv run python scripts/smoke_run.py data/golden/lecture.mp4 lecture
    uv run python scripts/smoke_run.py data/golden/vlog.mp4 vlog

목적: detector가 실 영상에서 크래시 없이 돌아가는지 빠르게 확인.
lecture 카테고리에선 script.md의 의도된 9개 이상 마커와 검출 결과 overlap 요약도 출력.
임계값 튜닝/평가는 별도 시스템 도입 후 진행 (이건 스모크일 뿐).
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import cast

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vidoctor.graph import run_analysis  # noqa: E402
from vidoctor.graph.state import (  # noqa: E402
    DIM_TO_STATE_FIELD,
    AnalysisState,
    Category,
)

# script.md 의도된 마커. 영상 길이가 의도(3:00)와 다를 경우 매칭이 흔들리는 것은 정상.
EXPECTED_LECTURE: list[dict] = [
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

DUMP_FIELDS = (
    "fillers",
    "cps_anomalies",
    "dead_zones",
    "gaze_issues",
    "content_gaps",
    "suggestions",
)

OVERLAP_TOL = 5.0


def overlaps(a_start: float, a_end: float, b_start: float, b_end: float, tol: float) -> bool:
    return (a_start - tol) <= b_end and (b_start - tol) <= a_end


def _events_dump(state: AnalysisState, field: str) -> list[dict]:
    events = state.get(field, []) or []  # type: ignore[literal-required]
    return [
        e.model_dump(mode="json") if isinstance(e, BaseModel) else dict(e) for e in events
    ]


def print_lecture_match_table(state: AnalysisState) -> None:
    print("\n=== expected vs detected (lecture markers) ===")
    print(f"{'dim':<12} {'expected':<14} {'hits':<6} note")
    print("-" * 70)
    for exp in EXPECTED_LECTURE:
        events = _events_dump(state, DIM_TO_STATE_FIELD[exp["dimension"]])
        hits = [
            ev for ev in events
            if overlaps(exp["start"], exp["end"], ev["start"], ev["end"], OVERLAP_TOL)
        ]
        rng = f"{int(exp['start'])}-{int(exp['end'])}s"
        mark = "✓" if hits else "✗"
        print(f"{exp['dimension']:<12} {rng:<14} {mark}{len(hits):<5} {exp['note']}")


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: smoke_run.py <video_path> <category>", file=sys.stderr)
        sys.exit(2)

    video_path = sys.argv[1]
    category = sys.argv[2]
    if not Path(video_path).exists():
        print(f"file not found: {video_path}", file=sys.stderr)
        sys.exit(2)

    print(f"[smoke] start  video={video_path} category={category}")
    t0 = time.time()
    state = asyncio.run(run_analysis(video_path, cast(Category, category)))
    elapsed = time.time() - t0
    print(f"[smoke] graph done in {elapsed:.1f}s")

    out_path = ROOT / "data" / "golden" / f"smoke_{Path(video_path).stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "video_path": video_path,
        "category": category,
        "elapsed_sec": round(elapsed, 1),
        "transcript_word_count": len(state.get("transcript") or []),
        **{f: _events_dump(state, f) for f in DUMP_FIELDS},
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[smoke] dumped → {out_path}")

    if category == "lecture":
        print_lecture_match_table(state)


if __name__ == "__main__":
    main()
