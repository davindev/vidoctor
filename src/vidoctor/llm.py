"""OpenAI LLM 래퍼 + Langfuse trace 통합 + 호출 비용·latency 측정 헬퍼.

- LangChain ChatOpenAI에 Langfuse callback을 부착해 모든 호출이 자동 trace.
- detect_content_gap (GPT-4o Vision), generate_suggestions (GPT-4o-mini)에서 공통 사용.
- Langfuse v4부터 글로벌 클라이언트 초기화 후 CallbackHandler가 그 상태를 공유하는 구조.
- LLMCallMetrics·estimate_cost_usd로 production·평가 양쪽이 동일 단가표 공유.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, TypeVar, cast

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel

from vidoctor.config import get_settings

# 모델별 1M 토큰당 USD 단가 (input, output). OpenAI 공식가. 캐시·discount 미반영.
# 정확 청구는 dashboard 확인. 새 모델 추가 시 여기 한 줄.
_PRICE_USD_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}

# step 추가 시 여기 한 줄 — 오타·불일치는 컴파일 타임에 검출.
LLMStep = Literal["content_gap", "suggestions", "classify_category"]


@dataclass(frozen=True)
class LLMCallMetrics:
    """LLM 1회 호출 메타."""

    step: LLMStep
    model: str
    cost_usd: float
    latency_sec: float
    prompt_tokens: int
    completion_tokens: int

    @classmethod
    def empty(cls, step: LLMStep, model: str) -> LLMCallMetrics:
        """LLM 호출이 생략된 경우(샘플 0건 등)의 zero metrics."""
        return cls(
            step=step,
            model=model,
            cost_usd=0.0,
            latency_sec=0.0,
            prompt_tokens=0,
            completion_tokens=0,
        )


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """단가표 기준 비용 추정. 모델이 표에 없으면 0.0."""
    if model not in _PRICE_USD_PER_1M:
        return 0.0
    in_rate, out_rate = _PRICE_USD_PER_1M[model]
    return (prompt_tokens / 1_000_000) * in_rate + (completion_tokens / 1_000_000) * out_rate


def extract_token_usage(raw: Any) -> tuple[int, int]:
    """LangChain AIMessage에서 (prompt_tokens, completion_tokens) 추출. 없으면 (0, 0)."""
    usage = getattr(raw, "usage_metadata", None) or {}
    return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))


_R = TypeVar("_R", bound=BaseModel)


async def invoke_structured_with_metrics(
    chat_model: ChatOpenAI,
    schema: type[_R],
    messages: list[BaseMessage],
    *,
    step: LLMStep,
) -> tuple[_R, LLMCallMetrics]:
    """structured output ainvoke + latency·token usage 측정 → (parsed, metrics).

    `with_structured_output(include_raw=True)`로 raw + parsed 둘 다 받아 raw에서
    token 메타를 추출하고, 호출 latency는 perf_counter로 측정. content_gap·suggestions
    가 동일 흐름이라 한 곳에 캡슐화.
    """
    structured = chat_model.with_structured_output(schema, include_raw=True)
    t0 = time.perf_counter()
    result = cast(dict, await structured.ainvoke(messages))
    latency = time.perf_counter() - t0

    raw = result["raw"]
    parsed = cast(_R, result["parsed"])
    prompt_tok, completion_tok = extract_token_usage(raw)
    model_name = cast(str, chat_model.model)
    metrics = LLMCallMetrics(
        step=step,
        model=model_name,
        cost_usd=estimate_cost_usd(model_name, prompt_tok, completion_tok),
        latency_sec=latency,
        prompt_tokens=prompt_tok,
        completion_tokens=completion_tok,
    )
    return parsed, metrics


@lru_cache(maxsize=1)
def _init_langfuse() -> Langfuse:
    """Langfuse 글로벌 클라이언트 초기화 (1회).

    이후 CallbackHandler 인스턴스들은 이 글로벌 상태를 공유.
    """
    settings = get_settings()
    return Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        host=settings.langfuse_host,
    )


@lru_cache(maxsize=1)
def _langfuse_handler() -> CallbackHandler:
    _init_langfuse()
    return CallbackHandler()


def get_chat_model(
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> ChatOpenAI:
    """LangChain ChatOpenAI 인스턴스. Langfuse callback 자동 부착.

    호출자는 표준 .invoke() / .ainvoke() / structured output 등 LangChain API 사용.
    기본은 비용 가벼운 gpt-4o-mini. content_gap 분석은 model="gpt-4o" 명시 필요.

    max_tokens는 structured output 길이 폭발(모델이 종료 신호 없이 list 항목을 무한
    생성해 length limit에 닿는 케이스)을 차단할 때 명시.
    """
    settings = get_settings()
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,  # type: ignore[call-arg]
        # 429 burst를 SDK 지수 backoff로 흡수. 5회면 burst-heavy eval 안정 통과,
        # 영구 실패 시 빠른 fail-fast가 가능한 균형점.
        max_retries=5,
        api_key=settings.openai_api_key,
        callbacks=[_langfuse_handler()],
    )
