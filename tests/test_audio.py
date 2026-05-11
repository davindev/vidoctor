"""실제 WhisperX 통합 테스트.

기본은 skip (모델 다운로드 + 추론에 수 분 소요).
실행: VIDOCTOR_RUN_INTEGRATION=1 uv run pytest tests/test_audio.py -v
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from vidoctor.audio.transcribe import transcribe_video

INTEGRATION_ENABLED = os.environ.get("VIDOCTOR_RUN_INTEGRATION") == "1"
SAY_AVAILABLE = shutil.which("say") is not None


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
