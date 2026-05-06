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


def _detector_node_name(dim: Dimension) -> str:
    """차원 이름 → graph 노드 이름. 한 곳에서 변환해 매핑 일관성 유지."""
    return f"detect_{dim}"


# Dimension → detector 함수 dict. 노드 등록·라우팅·fan-in이 모두 이 매핑에서 derive
# 되므로 신규 차원 추가 시 여기 한 줄 + state.py의 Dimension·DIM_TO_STATE_FIELD·
# CATEGORY_DIMENSIONS만 손보면 끝난다.
_DETECTORS: dict[Dimension, Callable[[AnalysisState], Awaitable[dict]]] = {
    "filler": detect_filler,
    "cps": detect_cps,
    "dead_zone": detect_dead_zone,
    "gaze": detect_gaze,
    "content_gap": detect_content_gap,
}

_ALL_DETECTOR_NODES: tuple[str, ...] = tuple(_detector_node_name(d) for d in _DETECTORS)


def _route_by_category(state: AnalysisState) -> list[str]:
    """transcribe 직후 카테고리를 보고 활성 detection 노드 이름을 반환.

    LangGraph add_conditional_edges는 list 반환 시 fan-out — 반환된 노드들이 동시 실행.
    비활성 차원은 노드 자체가 호출되지 않아 state에 키도 안 생긴다.
    """
    return [_detector_node_name(d) for d in CATEGORY_DIMENSIONS[state["category"]]]


def build_graph() -> CompiledStateGraph:
    """5차원 분석 파이프라인.

    구조:
        START → transcribe → (conditional fan-out, 카테고리별 활성 차원만)
                          → join → generate_suggestions → END

    카테고리별 활성 차원은 `CATEGORY_DIMENSIONS` 매핑이 single source of truth.
    """
    g: StateGraph = StateGraph(AnalysisState)

    g.add_node("transcribe", transcribe)
    for dim, fn in _DETECTORS.items():
        # LangGraph add_node의 StateNode 제네릭 추론이 dict value의 Callable과 매칭되지
        # 않아 ignore. 직접 호출(g.add_node("name", detect_fn))은 통과하지만 그 형태는
        # 5차원 SSOT 회복 의도와 어긋남.
        g.add_node(_detector_node_name(dim), fn)  # pyright: ignore[reportArgumentType]
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
