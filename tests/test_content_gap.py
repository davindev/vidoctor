"""Content gap 검출 테스트.

순수 함수(_transcript_around, _build_message) 단위 + 실 GPT-4o Vision 호출 통합.
통합 테스트는 VIDOCTOR_RUN_INTEGRATION=1로 활성 (LLM 비용·지연 발생).
"""

import os
from pathlib import Path

import numpy as np
import pytest

from tests._helpers import _w, write_video
from vidoctor.vision.content_gap import (
    MAX_SAMPLES,
    _build_message,
    _detect_scene_cuts,
    _FrameSample,
    _merge_sample_times,
    _transcript_around,
    _uniform_times,
    detect_content_gap_events,
)

INTEGRATION_ENABLED = os.environ.get("VIDOCTOR_RUN_INTEGRATION") == "1"


# ---------------------------------------------------------------------------
# _transcript_around
# ---------------------------------------------------------------------------


def test_transcript_around_picks_words_within_window():
    transcript = [
        _w("멀리", 0.0, 0.5),
        _w("가까이", 28.0, 28.5),
        _w("중심", 30.0, 30.5),
        _w("가까이", 32.0, 32.5),
        _w("멀리", 60.0, 60.5),
    ]
    text = _transcript_around(transcript, time_sec=30.0)
    # 30 ± 15 = 15~45 범위에 든 단어들만
    assert "가까이" in text
    assert "중심" in text
    assert "멀리" not in text


def test_transcript_around_empty_when_no_words_in_window():
    transcript = [_w("멀리", 0.0, 0.5)]
    text = _transcript_around(transcript, time_sec=60.0)
    assert text == ""


# ---------------------------------------------------------------------------
# _build_message
# ---------------------------------------------------------------------------


def test_build_message_contains_rubric_and_frames():
    samples = [
        _FrameSample(time_sec=10.0, image_b64="AAAA", transcript_text="안녕"),
        _FrameSample(time_sec=40.0, image_b64="BBBB", transcript_text=""),
    ]
    rubric = "rubric_marker_text"
    msg = _build_message(samples, rubric)

    content = msg.content
    assert isinstance(content, list)

    # rubric, transcript x2, image x2, 마무리 지시 = 6개 블록
    text_blocks = [
        b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"
    ]
    image_blocks = [
        b for b in content if isinstance(b, dict) and b.get("type") == "image_url"
    ]

    assert any("rubric_marker_text" in t for t in text_blocks)
    assert len(image_blocks) == 2
    # 빈 transcript는 명시 안내로 대체
    assert any("발화 없음" in t for t in text_blocks)


# ---------------------------------------------------------------------------
# _uniform_times / _merge_sample_times
# ---------------------------------------------------------------------------


def test_uniform_times_centers_first_sample():
    # SAMPLE_INTERVAL_SEC=30 가정. 첫 샘플은 윈도우 중앙 = 15
    times = _uniform_times(duration=180.0)
    assert times[0] == 15.0
    assert times[-1] < 180.0
    # 30s 간격
    diffs = [b - a for a, b in zip(times, times[1:], strict=False)]
    assert all(d == 30.0 for d in diffs)


def test_uniform_times_short_duration():
    # 5s 영상은 첫 샘플(15s)도 못 만듦
    assert _uniform_times(duration=5.0) == []


def test_merge_keeps_uniform_when_no_cuts():
    uniform = [15.0, 45.0, 75.0]
    assert _merge_sample_times(uniform, [], MAX_SAMPLES) == uniform


def test_merge_prefers_cut_over_close_uniform():
    # 컷이 균등 샘플과 SCENE_DEDUP_THRESHOLD 이내면 컷 우선 (정보 풍부)
    uniform = [15.0, 45.0, 75.0]
    cuts = [16.0, 60.0]  # 16은 15와 인접(<5)이지만 컷 우선
    merged = _merge_sample_times(uniform, cuts, MAX_SAMPLES)
    assert merged == [16.0, 45.0, 60.0, 75.0]


def test_merge_caps_at_max_samples():
    # 12개를 cap=10으로 줄임. 균등 분포 추출
    uniform = [float(i * 10) for i in range(12)]
    capped = _merge_sample_times(uniform, [], max_samples=10)
    assert len(capped) == 10
    assert capped[0] == uniform[0]


def test_merge_preserves_order():
    uniform = [60.0, 90.0]
    cuts = [10.0, 50.0]
    merged = _merge_sample_times(uniform, cuts, MAX_SAMPLES)
    assert merged == sorted(merged)


# ---------------------------------------------------------------------------
# _detect_scene_cuts (합성 영상)
# ---------------------------------------------------------------------------


def _make_video_with_cuts(path: Path, duration_sec: float = 9.0) -> None:
    """3초마다 색이 크게 바뀌는 영상 (3개 씬)."""
    colors = [(40, 40, 40), (240, 40, 40), (40, 240, 40)]

    def frame_fn(_i: int, t: float) -> np.ndarray:
        scene = min(int(t // 3), len(colors) - 1)
        return np.full((240, 320, 3), colors[scene], dtype=np.uint8)

    write_video(path, duration_sec, frame_fn)


def test_detect_scene_cuts_finds_color_changes(tmp_path: Path):
    video = tmp_path / "cuts.mp4"
    _make_video_with_cuts(video)
    cuts = _detect_scene_cuts(str(video))
    # 3·6초 부근에서 컷 발생 기대 (정확한 시각은 PySceneDetect 알고리즘 의존)
    assert any(2.5 <= c <= 4.0 for c in cuts), f"3s 부근 컷 못 잡음: {cuts}"


# ---------------------------------------------------------------------------
# 실 LLM 통합 (skip-by-default)
# ---------------------------------------------------------------------------


def _make_lecture_video(path: Path, duration_sec: float = 60.0) -> None:
    """단일 색 영상 — 실제 강의 콘텐츠 없음. LLM이 정보 부족으로 판정해야 정상."""
    blank = np.full((480, 640, 3), 200, dtype=np.uint8)
    write_video(path, duration_sec, lambda _i, _t: blank, size=(640, 480))


@pytest.mark.skipif(not INTEGRATION_ENABLED, reason="VIDOCTOR_RUN_INTEGRATION=1 필요")
async def test_content_gap_returns_response_for_blank_lecture(tmp_path: Path):
    video = tmp_path / "blank.mp4"
    _make_lecture_video(video, duration_sec=60.0)

    transcript = [_w("이건", 1.0, 1.4), _w("강의입니다", 1.5, 2.5)]
    events, _metrics = await detect_content_gap_events(str(video), transcript, "lecture")

    # 빈 슬라이드 + 짧은 발화 → 정보 부족 issue 1개+ 반환 기대 (LLM 판단)
    # 결과 형식만 검증 (구체 결정은 LLM이라 결정적이지 않음)
    for ev in events:
        assert ev.end >= ev.start
        assert ev.description


