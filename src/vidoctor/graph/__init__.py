"""LangGraph 기반 5차원 분석 오케스트레이션 패키지."""

from collections.abc import Callable
from functools import lru_cache
from typing import Any, cast

from vidoctor.graph.pipeline import build_graph
from vidoctor.graph.state import (
    AnalysisState,
    Category,
    ContentGapEvent,
    CPSEvent,
    DeadZoneEvent,
    FillerEvent,
    GazeEvent,
    Suggestion,
    Word,
)


# 그래프 구조는 불변 — 영상마다 재컴파일하지 않는다 (수십 ms 절감).
@lru_cache(maxsize=1)
def _compiled_graph() -> Any:
    return build_graph()


async def run_analysis(
    video_path: str,
    category: Category,
    on_node_complete: Callable[[str], None] | None = None,
) -> AnalysisState:
    """카테고리별 5차원 graph 실행. UI·스크립트 공용 진입점.

    `on_node_complete`: 노드 완료 시 노드 이름으로 호출되는 진행률 콜백 (동기·블로킹).
    멱등 가정 — LangGraph의 retry·fan-out·super-step에서 같은 노드 이름이 토폴로지
    순서와 다르게 다시 들어올 수 있다.

    astream 단일 패스로 updates(진행)와 values(최종 state)를 동시 회수.
    콜백 없으면 values만 구독해 update yield 비용 절약.
    """
    g = _compiled_graph()
    initial: dict[str, Any] = {"video_path": video_path, "category": category}
    final_state: dict[str, Any] | None = None
    stream_modes = ["updates", "values"] if on_node_complete else ["values"]

    async for mode, payload in g.astream(initial, stream_mode=stream_modes):
        if mode == "values":
            final_state = payload
        elif mode == "updates" and on_node_complete is not None:
            for node_name in payload:
                on_node_complete(node_name)

    if final_state is None:
        raise RuntimeError("graph가 최종 state를 yield하지 않았습니다")
    return cast(AnalysisState, final_state)


__all__ = [
    "AnalysisState",
    "CPSEvent",
    "Category",
    "ContentGapEvent",
    "DeadZoneEvent",
    "FillerEvent",
    "GazeEvent",
    "Suggestion",
    "Word",
    "build_graph",
    "run_analysis",
]
