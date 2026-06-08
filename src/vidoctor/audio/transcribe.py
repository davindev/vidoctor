"""WhisperX 기반 한국어 ASR + wav2vec2 forced alignment.

ASR 모델로 텍스트 추출 후 wav2vec2로 단어 단위 ±20ms 정렬. 모델은 첫 호출 시 lazy
load되어 프로세스 수명 동안 캐시. settings.whisper_model로 모델 swap (기본 large-v3-turbo).
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

from vidoctor.config import ROOT, get_settings
from vidoctor.graph.state import Word

# 기본은 KsponSpeech fine-tuned. ROOT 기준이라 로컬·Docker 양쪽에서 해석된다.
DEFAULT_MODEL_NAME = str(ROOT / "models" / "whisper-ko-ksponspeech-ct2")
LANGUAGE = "ko"


def _resolve_runtime() -> tuple[str, str, int]:
    """(device, compute_type, batch_size)를 런타임에 결정한다.

    VIDOCTOR_DEVICE=cuda면 Modal GPU, 그 외(로컬·빌드 prewarm)는 CPU.
    _load_models 첫 호출 시 한 번 평가된다.
    """
    if os.environ.get("VIDOCTOR_DEVICE") == "cuda":
        # 모델이 int8 양자화 저장이라 GPU도 int8로 맞춘다.
        return "cuda", "int8", 24
    return "cpu", "int8", 16


@dataclass(frozen=True)
class _LoadedModels:
    """lazy load된 WhisperX ASR + wav2vec2 align 모델 묶음 (lru_cache 결과)."""

    asr: Any
    align_model: Any
    align_metadata: Any
    device: str
    batch_size: int


@lru_cache(maxsize=1)
def _load_models() -> _LoadedModels:
    """ASR + wav2vec2 align 모델 lazy load. settings.whisper_model 우선, 없으면 default."""
    model_name = get_settings().whisper_model or DEFAULT_MODEL_NAME
    device, compute_type, batch_size = _resolve_runtime()
    asr = whisperx.load_model(
        model_name,
        device=device,
        compute_type=compute_type,
        language=LANGUAGE,
    )
    align_model, align_metadata = whisperx.load_align_model(
        language_code=LANGUAGE,
        device=device,
    )
    return _LoadedModels(
        asr=asr,
        align_model=align_model,
        align_metadata=align_metadata,
        device=device,
        batch_size=batch_size,
    )


def _transcribe_sync(media_path: str) -> tuple[list[Word], np.ndarray]:
    """파일 fast-fail 후 WhisperX ASR + wav2vec2 align → Word 리스트 + 16kHz mono."""
    if not Path(media_path).exists():
        raise FileNotFoundError(f"미디어 파일 없음: {media_path}")

    models = _load_models()
    audio = whisperx.load_audio(media_path)

    asr_result = models.asr.transcribe(
        audio, batch_size=models.batch_size, language=LANGUAGE
    )
    aligned = whisperx.align(
        asr_result["segments"],
        models.align_model,
        models.align_metadata,
        audio,
        models.device,
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
