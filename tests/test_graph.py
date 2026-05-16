"""Graph 토폴로지 테스트 — 카테고리별 활성 차원 + stream chunk 분기."""

import numpy as np
import pytest

from vidoctor.graph import build_graph
from vidoctor.graph.state import (
    CATEGORY_DIMENSIONS,
    DIM_TO_STATE_FIELD,
    Category,
    Dimension,
)
from vidoctor.llm import LLMCallMetrics


@pytest.fixture(autouse=True)
def _stub_heavy_nodes(monkeypatch):
    """그래프 토폴로지 테스트에선 미디어 파일·LLM·모델을 요구하는 노드를 우회.

    실제 ASR/SSIM 동작은 모듈 단위 통합 테스트(tests/test_audio.py, test_dead_zone.py)에서 검증.
    """

    async def _empty_words(_path: str):
        return [], np.array([], dtype=np.float32)

    async def _empty_dead_zone(_path: str, _category, *, audio=None):
        return []

    async def _empty_content_gap(_path: str, _transcript, _category):
        return [], LLMCallMetrics.empty(step="content_gap", model="gpt-4o")

    async def _empty_gaze(_path: str):
        return []

    monkeypatch.setattr("vidoctor.audio.transcribe.transcribe_video", _empty_words)
    monkeypatch.setattr(
        "vidoctor.vision.dead_zone.detect_dead_zone_events", _empty_dead_zone
    )
    monkeypatch.setattr(
        "vidoctor.vision.content_gap.detect_content_gap_events", _empty_content_gap
    )
    monkeypatch.setattr("vidoctor.vision.gaze.detect_gaze_events", _empty_gaze)


def test_graph_compiles():
    g = build_graph()
    assert g is not None


@pytest.mark.parametrize("category", ["lecture", "vlog", "other"])
async def test_graph_runs_with_active_dimensions(category: Category):
    g = build_graph()
    result = await g.ainvoke(
        {"video_path": "/tmp/x.mp4", "category": category},
    )
    base = {"transcript", "suggestions"}
    active_fields = {DIM_TO_STATE_FIELD[d] for d in CATEGORY_DIMENSIONS[category]}
    assert base.issubset(result.keys())
    assert active_fields.issubset(result.keys())


@pytest.mark.parametrize(
    "category,inactive_dims",
    [
        ("lecture", set[Dimension]()),
        ("vlog", {"gaze", "content_gap"}),
        ("other", {"gaze"}),
    ],
)
async def test_inactive_dimensions_are_not_in_state(
    category: Category, inactive_dims: set[Dimension]
):
    g = build_graph()
    result = await g.ainvoke(
        {"video_path": "/tmp/x.mp4", "category": category},
    )
    inactive_fields = {DIM_TO_STATE_FIELD[d] for d in inactive_dims}
    assert inactive_fields.isdisjoint(result.keys())


async def test_stream_yields_active_node_chunks():
    g = build_graph()
    nodes_seen: set[str] = set()
    async for chunk in g.astream(
        {"video_path": "/tmp/x.mp4", "category": "vlog"},
    ):
        nodes_seen.update(chunk.keys())
    assert "transcribe" in nodes_seen
    assert "generate_suggestions" in nodes_seen
    assert {"detect_filler", "detect_cps", "detect_dead_zone"} <= nodes_seen
    assert "detect_gaze" not in nodes_seen
    assert "detect_content_gap" not in nodes_seen
