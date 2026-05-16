"""ASR transcribe 테스트 — 파일 부재 fail-fast 단위 + 실 WhisperX 통합.

통합 테스트는 모델 다운로드 + 추론에 수 분 소요. 기본 skip.
실행: VIDOCTOR_RUN_INTEGRATION=1 uv run pytest tests/test_audio.py -v
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from vidoctor.audio.transcribe import _transcribe_sync, transcribe_video

INTEGRATION_ENABLED = os.environ.get("VIDOCTOR_RUN_INTEGRATION") == "1"
SAY_AVAILABLE = shutil.which("say") is not None


# ---------------------------------------------------------------------------
# 파일 부재 fail-fast — 비싼 WhisperX 로드 전에 즉시 abort 회귀 가드
# ---------------------------------------------------------------------------


def test_transcribe_sync_raises_on_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="미디어 파일 없음"):
        _transcribe_sync(str(tmp_path / "nope.mp4"))


async def test_transcribe_video_propagates_missing_file_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="미디어 파일 없음"):
        await transcribe_video(str(tmp_path / "nope.mp4"))


# ---------------------------------------------------------------------------
# 실 WhisperX 통합 (skip-by-default)
# ---------------------------------------------------------------------------


@pytest.fixture
def korean_sample_audio(tmp_path: Path) -> Path:
    """macOS say로 한국어 4초 샘플 생성."""
    if not SAY_AVAILABLE:
        pytest.skip("macOS say 명령 필요")
    out = tmp_path / "sample.aiff"
    subprocess.run(
        [
            "say",
            "-v",
            "Yuna",
            "-r",
            "200",
            "-o",
            str(out),
            "안녕하세요. 오늘은 타입스크립트 제네릭 입문 강의를 시작하겠습니다.",
        ],
        check=True,
    )
    return out


@pytest.mark.skipif(not INTEGRATION_ENABLED, reason="VIDOCTOR_RUN_INTEGRATION=1 필요")
async def test_transcribe_korean_sample(korean_sample_audio: Path):
    words, _ = await transcribe_video(str(korean_sample_audio))

    assert len(words) > 0, "단어가 하나도 추출되지 않음"
    assert any("안녕" in w.text for w in words), "'안녕'이 포함된 단어 없음"

    for w in words:
        assert w.end > w.start, f"end({w.end}) <= start({w.start})"
        assert w.start >= 0
        if w.score is not None:
            assert 0.0 <= w.score <= 1.0
