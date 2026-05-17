"""영상 → 카테고리 자동 분류.

분석 파이프라인 분기(`graph/pipeline.py` `CATEGORY_DIMENSIONS`)는 lecture / vlog / other
중 어떤 차원을 활성화할지 결정하므로 분석 시작 *직전*에 카테고리가 확정돼 있어야 한다.
이 모듈은 영상 시작/중간/후반 3시점 프레임을 gpt-4o-mini Vision에 보내 카테고리를 결정한다.

비용·시간 추정: 480p 3장 + gpt-4o-mini ≈ $0.0005, ~1-2초. 전체 파이프라인(수십 초)
대비 무시할 수준.
"""

from __future__ import annotations

import asyncio
import logging

import cv2
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from vidoctor.graph.state import Category
from vidoctor.llm import (
    LLMCallMetrics,
    get_chat_model,
    invoke_structured_with_metrics,
)
from vidoctor.vision._capture import encode_frame_jpeg, open_capture

_log = logging.getLogger(__name__)

_MODEL = "gpt-4o-mini"

# 분류엔 720p 불필요 — 480p로도 충분히 식별 가능하고 token 비용 더 절감.
_MAX_FRAME_HEIGHT = 480
_JPEG_QUALITY = 75

# 시작 직후 / 중간 / 후반 — 인트로/엔딩 컷어웨이가 본문과 다른 케이스를 흡수.
_SAMPLE_FRACTIONS = (0.05, 0.5, 0.8)

# confidence 임계. 미만이면 분기 안전한 'other'로 fallback.
_MIN_CONFIDENCE = 0.5


class _CategoryDecision(BaseModel):
    category: Category = Field(description="lecture / vlog / other 중 하나")
    confidence: float = Field(
        description="0.0-1.0. 명확하면 0.8+, 애매하면 0.5 미만으로 답할 것."
    )


def _extract_frames_sync(video_path: str) -> list[str]:
    """3개 분위 시각에서 프레임 추출 → base64 jpg 리스트. 실패 프레임은 skip."""
    with open_capture(video_path) as cap:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0
        if duration <= 0:
            return []
        images: list[str] = []
        for frac in _SAMPLE_FRACTIONS:
            cap.set(cv2.CAP_PROP_POS_MSEC, duration * frac * 1000.0)
            ret, frame = cap.read()
            if not ret:
                continue
            images.append(
                encode_frame_jpeg(frame, max_height=_MAX_FRAME_HEIGHT, quality=_JPEG_QUALITY)
            )
        return images


_PROMPT = """당신은 영상 카테고리 분류기입니다. 아래 영상 프레임들을 보고 카테고리를 \
다음 중 하나로 결정하세요.

- lecture: 강의·강연·발표·튜토리얼. 슬라이드/화이트보드/한 명의 화자가 정면 설명, \
  교육적·정보 전달 목적이 분명한 영상.
- vlog: 일상 기록·개인 브이로그·여행·먹방·운동기록. 자유로운 환경, 빠른 컷, \
  편집 자막/이모지가 많은 영상.
- other: lecture·vlog 어느 쪽도 아닌 경우 (인터뷰, 음악, 게임, 시연, 광고 등).

confidence는 0.0-1.0. 명확히 분류 가능하면 0.8 이상, 애매하면 0.5 미만으로 답하라."""


async def classify_category(video_path: str) -> tuple[Category, LLMCallMetrics]:
    """영상 → (카테고리, LLM 호출 메타). 분류 자체가 실패해도 분석 전체가 죽지 않도록
    프레임 추출 실패·LLM 호출 실패·자신감 부족 모두 'other'로 안전하게 fallback."""
    empty = LLMCallMetrics.empty("classify_category", _MODEL)
    images = await asyncio.to_thread(_extract_frames_sync, video_path)
    if not images:
        return "other", empty

    content: list[str | dict] = [{"type": "text", "text": _PROMPT}]
    for b64 in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )
    message = HumanMessage(content=content)

    model = get_chat_model(model=_MODEL, temperature=0.0, max_tokens=128)
    try:
        parsed, metrics = await invoke_structured_with_metrics(
            model, _CategoryDecision, [message], step="classify_category"
        )
    except Exception:
        # 사용자가 명시적으로 카테고리 골랐으면 성공했을 분석을 자동 분류 LLM 장애가
        # 죽이지 않도록 — fallback 후 분석은 진행.
        _log.warning("classify_category LLM 호출 실패, 'other'로 fallback", exc_info=True)
        return "other", empty

    if parsed.confidence >= _MIN_CONFIDENCE:
        return parsed.category, metrics
    return "other", metrics
