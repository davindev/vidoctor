"""5차원 분석 graph 구성 — START → transcribe → 카테고리별 fan-out → suggestions → END."""

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
    AnalysisState,
    Dimension,
)


def detector_node_name(dim: Dimension) -> str:
    """차원 이름 → graph 노드 이름. 매핑 일관성 위한 단일 변환점."""
    return f"detect_{dim}"


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
        g.add_node(detector_node_name(dim), fn)  # pyright: ignore[reportArgumentType]
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
