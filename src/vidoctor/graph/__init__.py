from typing import cast

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


async def run_analysis(video_path: str, category: Category) -> AnalysisState:
    """카테고리별 5차원 graph 실행. UI·스크립트 공용 진입점.

    LangGraph `ainvoke`는 untyped `dict`를 돌려주지만 우리 그래프 schema는 AnalysisState
    TypedDict로 고정 — 한 곳에서 cast해 호출자 부담 제거.
    """
    result = await build_graph().ainvoke({"video_path": video_path, "category": category})
    return cast(AnalysisState, result)


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
