import pytest

from vidoctor.graph import build_graph
from vidoctor.graph.state import Category


@pytest.fixture(autouse=True)
def _stub_transcribe(monkeypatch):
    """그래프 토폴로지 테스트에선 실제 WhisperX 호출을 우회.

    transcribe 노드가 미디어 파일을 요구하지 않도록 transcribe_video를 빈 리스트로 대체.
    실제 ASR 동작은 tests/test_audio.py(통합 테스트)에서 검증.
    """

    async def _empty(_path: str):
        return []

    monkeypatch.setattr("vidoctor.audio.transcribe.transcribe_video", _empty)


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
