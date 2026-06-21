"""5차원 분석 graph 구성 — START → transcribe → 카테고리별 fan-out → suggestions → END."""

import functools
import logging
from collections.abc import Awaitable, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from vidoctor.graph.nodes import (
    detect_content_gap,
    detect_cps,
    detect_dead_zone,
    detect_filler,
    detect_gaze,
    generate_suggestions,
    transcribe,
)
from vidoctor.graph.state import (
    CATEGORY_DIMENSIONS,
    DIM_TO_STATE_FIELD,
    AnalysisState,
    Dimension,
)

_log = logging.getLogger(__name__)


def detector_node_name(dim: Dimension) -> str:
    """차원 이름 → graph 노드 이름. 매핑 일관성 위한 단일 변환점."""
    return f"detect_{dim}"


def _isolate(
    dim: Dimension, fn: Callable[[AnalysisState], Awaitable[dict]]
) -> Callable[[AnalysisState], Awaitable[dict]]:
    """detector 노드를 감싸 예외를 차원 단위로 격리한다.

    한 차원의 예외가 run 전체를 죽여 이미 검출된 다른 차원까지 폐기하는 것을 막는다.
    실패 차원은 빈 결과로 두고 failed_dimensions에 기록 — UI가 '검출 0건'과 구분해
    '분석 실패'로 표시할 수 있다.
    """
    field = DIM_TO_STATE_FIELD[dim]

    @functools.wraps(fn)
    async def wrapped(state: AnalysisState) -> dict:
        try:
            return await fn(state)
        except Exception:
            _log.exception("차원 검출 실패 — 격리하고 진행", extra={"dimension": dim})
            return {field: [], "failed_dimensions": [dim]}

    return wrapped


# Dimension → detector. 신규 차원: 이 dict + state.py(Dimension/DIM_TO_STATE_FIELD/CATEGORY_DIMENSIONS).
_DETECTORS: dict[Dimension, Callable[[AnalysisState], Awaitable[dict]]] = {
    "filler": detect_filler,
    "cps": detect_cps,
    "dead_zone": detect_dead_zone,
    "gaze": detect_gaze,
    "content_gap": detect_content_gap,
}

_ALL_DETECTOR_NODES: tuple[str, ...] = tuple(detector_node_name(d) for d in _DETECTORS)


def _route_by_category(state: AnalysisState) -> list[str]:
    """카테고리 → 활성 detection 노드 이름 리스트.

    LangGraph는 list 반환을 fan-out (동시 실행)으로 처리. 비활성 차원은 호출 안 됨.
    """
    return [detector_node_name(d) for d in CATEGORY_DIMENSIONS[state["category"]]]


def build_graph() -> CompiledStateGraph:
    """5차원 분석 파이프라인.

    구조:
        START → transcribe → (conditional fan-out, 카테고리별 활성 차원만)
                          → join → generate_suggestions → END
    """
    g: StateGraph = StateGraph(AnalysisState)

    g.add_node("transcribe", transcribe)
    for dim, fn in _DETECTORS.items():
        # StateNode 제네릭이 dict value Callable 추론 실패. 직접 호출하면 통과하나 SSOT 깨짐.
        g.add_node(detector_node_name(dim), _isolate(dim, fn))  # pyright: ignore[reportArgumentType]
    g.add_node("generate_suggestions", generate_suggestions)

    g.add_edge(START, "transcribe")
    g.add_conditional_edges(
        "transcribe",
        _route_by_category,
        {node: node for node in _ALL_DETECTOR_NODES},
    )
    for node in _ALL_DETECTOR_NODES:
        g.add_edge(node, "generate_suggestions")
    g.add_edge("generate_suggestions", END)

    return g.compile()
