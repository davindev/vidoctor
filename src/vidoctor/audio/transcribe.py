"""WhisperX 기반 한국어 ASR + wav2vec2 forced alignment.

faster-whisper-large-v3-turbo로 텍스트 추출 후 wav2vec2로 단어 단위 ±20ms 정렬.
모델은 첫 호출 시 lazy load되어 프로세스 수명 동안 캐시.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import whisperx

from vidoctor.graph.state import Word

DEVICE = "cpu"
COMPUTE_TYPE = "int8"
MODEL_NAME = "large-v3-turbo"
LANGUAGE = "ko"
BATCH_SIZE = 16


@dataclass(frozen=True)
class _LoadedModels:
    asr: Any
    align_model: Any
    align_metadata: Any


@lru_cache(maxsize=1)
def _load_models() -> _LoadedModels:
    asr = whisperx.load_model(
        MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        language=LANGUAGE,
    )
    align_model, align_metadata = whisperx.load_align_model(
        language_code=LANGUAGE,
        device=DEVICE,
    )
    return _LoadedModels(asr=asr, align_model=align_model, align_metadata=align_metadata)


def _transcribe_sync(media_path: str) -> list[Word]:
    if not Path(media_path).exists():
        raise FileNotFoundError(f"미디어 파일 없음: {media_path}")

    models = _load_models()
    audio = whisperx.load_audio(media_path)

    asr_result = models.asr.transcribe(audio, batch_size=BATCH_SIZE, language=LANGUAGE)
    aligned = whisperx.align(
        asr_result["segments"],
        models.align_model,
        models.align_metadata,
        audio,
        DEVICE,
        return_char_alignments=False,
    )

    words: list[Word] = []
    for segment in aligned.get("segments", []):
        for w in segment.get("words", []):
            text = w.get("word", "").strip()
            start = w.get("start")
            end = w.get("end")
            if start is None or end is None or not text:
                continue
            score = w.get("score")
            words.append(
                Word(
                    text=text,
                    start=float(start),
                    end=float(end),
                    score=float(score) if score is not None else None,
                )
            )
    return words


async def transcribe_video(media_path: str) -> list[Word]:
    """영상/오디오 파일 → 단어 단위 transcript.

    WhisperX 호출은 sync·CPU bound이라 to_thread로 분리해 LangGraph 이벤트 루프 차단 방지.
    """
    return await asyncio.to_thread(_transcribe_sync, media_path)
