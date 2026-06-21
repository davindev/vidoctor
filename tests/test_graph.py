"""Graph 토폴로지 테스트 — 카테고리별 활성 차원 + stream chunk 분기."""

import numpy as np
import pytest

from vidoctor.errors import SafeError
from vidoctor.graph import build_graph
from vidoctor.graph.state import (
    CATEGORY_DIMENSIONS,
    DIM_TO_STATE_FIELD,
    Category,
    Dimension,
    Word,
)
from vidoctor.llm import LLMCallMetrics


@pytest.fixture(autouse=True)
def _stub_heavy_nodes(monkeypatch):
    """그래프 토폴로지 테스트에선 미디어 파일·LLM·모델을 요구하는 노드를 우회.

    실제 ASR/SSIM 동작은 모듈 단위 통합 테스트(tests/test_audio.py, test_dead_zone.py)에서 검증.
    """

    async def _dummy_words(_path: str):
        # 빈 transcript는 transcribe 게이트에서 SafeError로 막히므로 최소 단어 1개를 둔다.
        # cps는 아래에서 stub하므로 audio 내용은 무관.
        return [Word(text="테스트", start=1.0, end=1.4)], np.array([], dtype=np.float32)

    async def _ko_language(_audio):
        return "ko", 0.99

    async def _empty_dead_zone(_path: str, _category, *, audio=None):
        return []

    async def _empty_content_gap(_path: str, _transcript, _category):
        return [], LLMCallMetrics.empty(step="content_gap", model="gpt-4o")

    async def _empty_gaze(_path: str):
        return []

    monkeypatch.setattr("vidoctor.audio.transcribe.transcribe_video", _dummy_words)
    monkeypatch.setattr("vidoctor.audio.transcribe.detect_language", _ko_language)
    # cps는 dummy transcript/audio로 실제 F0 추출을 돌리지 않도록 stub (토폴로지 검증 목적).
    monkeypatch.setattr("vidoctor.audio.cps.detect_cps_anomalies", lambda *a, **k: [])
    monkeypatch.setattr(
        "vidoctor.audio.cps.detect_cps_with_audio", lambda *a, **k: []
    )
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


async def test_empty_transcript_raises_no_speech(monkeypatch):
    """음성이 없으면(빈 transcript) 빈 결과로 완료하지 않고 SafeError로 중단한다."""

    async def _no_words(_path: str):
        return [], np.array([], dtype=np.float32)

    monkeypatch.setattr("vidoctor.audio.transcribe.transcribe_video", _no_words)

    g = build_graph()
    with pytest.raises(SafeError):
        await g.ainvoke({"video_path": "/tmp/x.mp4", "category": "lecture"})


async def test_foreign_language_is_rejected(monkeypatch):
    """비한국어로 확신하면(prob≥0.5) 전사 전에 SafeError로 중단한다."""

    async def _english(_audio):
        return "en", 0.95

    monkeypatch.setattr("vidoctor.audio.transcribe.detect_language", _english)

    g = build_graph()
    with pytest.raises(SafeError):
        await g.ainvoke({"video_path": "/tmp/x.mp4", "category": "lecture"})


async def test_low_confidence_language_passes(monkeypatch):
    """비한국어여도 확신도가 낮으면(<0.5) 거부하지 않고 진행한다(음악·모호 입력 통과)."""

    async def _uncertain(_audio):
        return "en", 0.3

    monkeypatch.setattr("vidoctor.audio.transcribe.detect_language", _uncertain)

    g = build_graph()
    result = await g.ainvoke({"video_path": "/tmp/x.mp4", "category": "lecture"})
    assert "transcript" in result  # 게이트 통과 → 정상 진행


async def test_detector_failure_is_isolated(monkeypatch):
    """한 차원 detector가 던져도 run이 죽지 않고, 나머지 차원·제안은 보존된다."""

    async def _boom_gaze(_path: str):
        raise RuntimeError("MediaPipe 폭발")

    monkeypatch.setattr("vidoctor.vision.gaze.detect_gaze_events", _boom_gaze)

    g = build_graph()
    result = await g.ainvoke({"video_path": "/tmp/x.mp4", "category": "lecture"})

    # gaze는 실패로 격리되고, 나머지 차원 필드와 제안은 정상 생성.
    assert result["failed_dimensions"] == ["gaze"]
    assert result["gaze_issues"] == []
    assert "suggestions" in result
    for dim in ("filler", "cps", "dead_zone", "content_gap"):
        assert DIM_TO_STATE_FIELD[dim] in result


async def test_suggestions_failure_preserves_findings(monkeypatch):
    """제안 생성이 던져도 검출 결과는 유실되지 않는다."""

    async def _boom_suggestions(_state):
        raise RuntimeError("제안 LLM 폭발")

    monkeypatch.setattr("vidoctor.suggestions.build_suggestions", _boom_suggestions)

    g = build_graph()
    result = await g.ainvoke({"video_path": "/tmp/x.mp4", "category": "vlog"})

    assert result["suggestions"] == []
    for dim in ("filler", "cps", "dead_zone"):
        assert DIM_TO_STATE_FIELD[dim] in result


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
