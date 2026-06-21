"""WhisperX 기반 한국어 ASR + wav2vec2 forced alignment.

ASR 모델로 텍스트 추출 후 wav2vec2로 단어 단위 정렬. 모델은 첫 호출 시 lazy
load되어 프로세스 수명 동안 캐시. settings.whisper_model로 모델 swap (기본 KsponSpeech).
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import whisperx

from vidoctor.config import ROOT, get_settings
from vidoctor.graph.state import Word

_log = logging.getLogger(__name__)

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
        # batch는 T4(16GB) OOM 여유를 위해 작게.
        return "cuda", "int8", 8
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


def _free_gpu_memory() -> None:
    """warm 재사용 컨테이너에서 분석 간 torch 캐시 누적을 끊는다.

    ct2 ASR은 torch와 별개 allocator라 torch 캐시를 비워야 ct2가 쓸 여유가 생긴다.
    단일 분석 peak(batch_size)와는 무관 — 누적 OOM 방지용.
    """
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


def _is_cuda_oom(exc: BaseException) -> bool:
    """ct2가 OOM을 타입 없는 RuntimeError로 던져 메시지로만 판별 가능하다.

    torch의 OutOfMemoryError도 RuntimeError 하위라 같은 분기로 걸린다.
    """
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _transcribe_sync(
    media_path: str, batch_size: int | None = None
) -> tuple[list[Word], np.ndarray]:
    """파일 fast-fail 후 WhisperX ASR + wav2vec2 align → Word 리스트 + 16kHz mono.

    batch_size override는 OOM 재시도에서 peak를 낮추기 위한 것.
    """
    if not Path(media_path).exists():
        raise FileNotFoundError(f"미디어 파일 없음: {media_path}")

    models = _load_models()
    audio = whisperx.load_audio(media_path)
    bs = batch_size if batch_size is not None else models.batch_size

    try:
        asr_result = models.asr.transcribe(audio, batch_size=bs, language=LANGUAGE)
        aligned = whisperx.align(
            asr_result["segments"],
            models.align_model,
            models.align_metadata,
            audio,
            models.device,
            return_char_alignments=False,
        )
    finally:
        # OOM 실패 시에도 정리돼야 재시도가 깨끗한 메모리에서 시작한다.
        _free_gpu_memory()

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

    OOM은 batch를 절반으로 줄여 1회 재시도한다 — 같은 batch 재시도는 peak가 같아 무의미함.
    """
    try:
        return await asyncio.to_thread(_transcribe_sync, media_path)
    except RuntimeError as e:
        if not _is_cuda_oom(e):
            raise
        retry_bs = max(1, _load_models().batch_size // 2)
        _log.warning("ASR GPU OOM — batch_size=%d로 낮춰 1회 재시도", retry_bs)
        return await asyncio.to_thread(_transcribe_sync, media_path, retry_bs)


# 언어 감지에는 본 ASR(KsponSpeech)이 아니라 작은 다국어 base 모델을 쓴다 — 한국어 특화
# fine-tuning은 다국어 판별 정확도를 잃어 감지에 부적합하다. tiny는 가볍고 감지엔 충분하다.
LANG_DETECT_MODEL = "tiny"


@lru_cache(maxsize=1)
def _load_lang_model() -> Any:
    """언어 감지 전용 tiny 다국어 Whisper. 본 ASR과 별개로 lazy load·캐시."""
    from faster_whisper import WhisperModel

    device, compute_type, _ = _resolve_runtime()
    return WhisperModel(LANG_DETECT_MODEL, device=device, compute_type=compute_type)


async def detect_language(audio: np.ndarray) -> tuple[str, float]:
    """16kHz mono 신호의 언어 (ISO 코드, 확신도) 추정. tiny 다국어 모델 기준.

    transcribe가 디코딩한 audio를 재사용해 ffmpeg 디코드를 다시 하지 않는다. tiny의
    detect_language는 신호 앞 30초만 보므로 가볍다. 호출자가 확신도 임계로 거부를 결정한다.
    """

    def _run() -> tuple[str, float]:
        lang, prob, _ = _load_lang_model().detect_language(audio)
        return lang, float(prob)

    return await asyncio.to_thread(_run)
