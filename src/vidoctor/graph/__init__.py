from collections.abc import Callable
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


async def run_analysis(
    video_path: str,
    category: Category,
    on_node_complete: Callable[[str], None] | None = None,
) -> AnalysisState:
    """카테고리별 5차원 graph 실행. UI·스크립트 공용 진입점.

    `on_node_complete`가 주어지면 노드 완료마다 노드 이름으로 호출된다 — Streamlit
    `st.status` 진행률 표시용. 호출자는 *멱등성*을 가정해야 한다 (LangGraph가 retry·
    super-step에서 같은 노드를 다시 yield할 수 있고, fan-out 노드는 dict 순회 순서로
    들어와 그래프 토폴로지 순서와 다를 수 있음). 콜백은 동기·블로킹.

    `astream`을 단일 패스로 사용해 진행 이벤트(updates)와 누적 최종 state(values)를
    같이 회수 — `ainvoke` 별도 호출로 두 번 돌리는 비용 회피. 콜백이 없으면 values만
    구독해 update yield 비용도 절약.
    """
    g = build_graph()
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
