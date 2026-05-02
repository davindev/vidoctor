from typing import Any

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


async def run_analysis(video_path: str, category: Category) -> dict[str, Any]:
    """카테고리별 5차원 graph 실행. UI·스크립트 공용 진입점."""
    return await build_graph().ainvoke({"video_path": video_path, "category": category})


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
