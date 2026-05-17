"""WhisperX 기반 한국어 ASR + wav2vec2 forced alignment.

ASR 모델로 텍스트 추출 후 wav2vec2로 단어 단위 ±20ms 정렬. 모델은 첫 호출 시 lazy
load되어 프로세스 수명 동안 캐시.

`VIDOCTOR_WHISPER_MODEL` 환경변수로 모델 path 또는 HF id 지정 가능. 기본은 OpenAI
영어 우세 large-v3-turbo이고, 한국어 fine-tuned 모델(예: ct2 변환된 KsponSpeech 학습)
경로를 지정하면 그쪽으로 swap.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import whisperx

from vidoctor.graph.state import Word

DEVICE = "cpu"                          # CPU 추론 (Apple Silicon에서 안정, CUDA 없는 환경 호환)
COMPUTE_TYPE = "int8"                   # int8 양자화 (속도 ↑ 메모리 ↓, 한국어 정확도 영향 미미)
DEFAULT_MODEL_NAME = "large-v3-turbo"
LANGUAGE = "ko"
BATCH_SIZE = 16                         # WhisperX 권장 default, 16GB RAM에서 안정


@dataclass(frozen=True)
class _LoadedModels:
    """lazy load된 WhisperX ASR + wav2vec2 align 모델 묶음 (lru_cache 결과)."""

    asr: Any
    align_model: Any
    align_metadata: Any


@lru_cache(maxsize=1)
def _load_models() -> _LoadedModels:
    asr = whisperx.load_model(
        os.environ.get("VIDOCTOR_WHISPER_MODEL", DEFAULT_MODEL_NAME),
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        language=LANGUAGE,
    )
    align_model, align_metadata = whisperx.load_align_model(
        language_code=LANGUAGE,
        device=DEVICE,
    )
    return _LoadedModels(asr=asr, align_model=align_model, align_metadata=align_metadata)


def _transcribe_sync(media_path: str) -> tuple[list[Word], np.ndarray]:
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
    return words, audio


async def transcribe_video(media_path: str) -> tuple[list[Word], np.ndarray]:
    """영상/오디오 파일 → (단어 단위 transcript, 16kHz mono ndarray).

    audio는 WhisperX가 이미 디코딩한 16kHz mono float32라 dead_zone VAD가 재사용 → ffmpeg
    호출 1회 절감. WhisperX 호출은 sync·CPU bound이라 to_thread로 이벤트 루프 차단 방지.
    """
    return await asyncio.to_thread(_transcribe_sync, media_path)
