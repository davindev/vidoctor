import pytest

from vidoctor.graph import build_graph
from vidoctor.graph.state import Category


@pytest.fixture(autouse=True)
def _stub_heavy_nodes(monkeypatch):
    """그래프 토폴로지 테스트에선 미디어 파일·LLM·모델을 요구하는 노드를 우회.

    실제 ASR/SSIM 동작은 모듈 단위 통합 테스트(tests/test_audio.py, test_dead_zone.py)에서 검증.
    """

    async def _empty_words(_path: str):
        return []

    async def _empty_events(_path: str, _transcript, _category):
        return []

    monkeypatch.setattr("vidoctor.audio.transcribe.transcribe_video", _empty_words)
    monkeypatch.setattr(
        "vidoctor.vision.dead_zone.detect_dead_zone_events", _empty_events
    )
    monkeypatch.setattr(
        "vidoctor.vision.content_gap.detect_content_gap_events", _empty_events
    )


def test_graph_compiles():
    g = build_graph()
    assert g is not None


@pytest.mark.parametrize("category", ["lecture", "vlog", "other"])
async def test_graph_runs_for_category(category: Category):
    g = build_graph()
    result = await g.ainvoke(
        {"video_path": "/tmp/x.mp4", "category": category},
    )
    expected = {
        "transcript",
        "fillers",
        "cps_anomalies",
        "dead_zones",
        "gaze_issues",
        "content_gaps",
        "suggestions",
    }
    assert expected.issubset(result.keys())


async def test_stream_yields_node_chunks():
    g = build_graph()
    nodes_seen: set[str] = set()
    async for chunk in g.astream(
        {"video_path": "/tmp/x.mp4", "category": "lecture"},
    ):
        nodes_seen.update(chunk.keys())
    assert "transcribe" in nodes_seen
    assert "generate_suggestions" in nodes_seen
    assert {"detect_filler", "detect_cps", "detect_dead_zone"} <= nodes_seen
