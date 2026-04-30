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
from vidoctor.graph.state import AnalysisState

DETECTION_NODES = (
    "detect_filler",
    "detect_cps",
    "detect_dead_zone",
    "detect_gaze",
    "detect_content_gap",
)


def build_graph() -> CompiledStateGraph:
    """5차원 분석 파이프라인.

    구조:
        START → transcribe → fan-out 5 detection → fan-in suggestions → END

    카테고리별 활성화는 각 detection 노드 내부에서 early-return으로 처리.
    노드 단위 trace를 위해 여러 노드로 분리 (LangGraph 자동 병렬 실행).
    """
    g: StateGraph = StateGraph(AnalysisState)

    g.add_node("transcribe", transcribe)
    g.add_node("detect_filler", detect_filler)
    g.add_node("detect_cps", detect_cps)
    g.add_node("detect_dead_zone", detect_dead_zone)
    g.add_node("detect_gaze", detect_gaze)
    g.add_node("detect_content_gap", detect_content_gap)
    g.add_node("generate_suggestions", generate_suggestions)

    g.add_edge(START, "transcribe")
    for node in DETECTION_NODES:
        g.add_edge("transcribe", node)
        g.add_edge(node, "generate_suggestions")
    g.add_edge("generate_suggestions", END)

    return g.compile()
