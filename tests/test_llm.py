"""LLM 래퍼 테스트 — 모델 인스턴스 + 비용·토큰 추출 순수 함수.

실 OpenAI 호출 + Langfuse trace 검증은 VIDOCTOR_RUN_INTEGRATION=1로 활성.
"""

import os

import pytest
from langchain_openai import ChatOpenAI

from vidoctor.llm import (
    LLMCallMetrics,
    estimate_cost_usd,
    extract_token_usage,
    get_chat_model,
)

INTEGRATION_ENABLED = os.environ.get("VIDOCTOR_RUN_INTEGRATION") == "1"


# ---------------------------------------------------------------------------
# get_chat_model — ChatOpenAI 인스턴스
# ---------------------------------------------------------------------------


def test_default_model_is_gpt_4o_mini():
    model = get_chat_model()
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-4o-mini"
    assert model.temperature == 0.0


def test_custom_model_and_temperature():
    model = get_chat_model(model="gpt-4o", temperature=0.3)
    assert model.model_name == "gpt-4o"
    assert model.temperature == 0.3


def test_langfuse_callback_attached():
    model = get_chat_model()
    # callbacks는 list 또는 BaseCallbackManager. None이 아니면 부착됨
    assert model.callbacks is not None


# ---------------------------------------------------------------------------
# estimate_cost_usd — 단가표 기반 비용 계산
# ---------------------------------------------------------------------------


def test_estimate_cost_gpt_4o_full_million():
    # 1M input × $2.50 + 1M output × $10.00 = $12.50
    assert estimate_cost_usd("gpt-4o", 1_000_000, 1_000_000) == pytest.approx(12.50)


def test_estimate_cost_gpt_4o_mini_full_million():
    # 1M input × $0.15 + 1M output × $0.60 = $0.75
    assert estimate_cost_usd("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)


def test_estimate_cost_unknown_model_returns_zero():
    # 단가표 미등록 모델은 비용 0 (silent fallback, 단가 누락 시 0 청구로 가시화).
    assert estimate_cost_usd("nonexistent", 1_000_000, 1_000_000) == 0.0


def test_estimate_cost_zero_tokens():
    assert estimate_cost_usd("gpt-4o", 0, 0) == 0.0


# ---------------------------------------------------------------------------
# extract_token_usage — LangChain AIMessage usage_metadata 추출
# ---------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, usage_metadata: dict | None = None) -> None:
        if usage_metadata is not None:
            self.usage_metadata = usage_metadata


def test_extract_token_usage_with_metadata():
    msg = _FakeMessage({"input_tokens": 100, "output_tokens": 50})
    assert extract_token_usage(msg) == (100, 50)


def test_extract_token_usage_missing_attribute_returns_zero():
    # usage_metadata 속성 자체가 없는 경우 (구버전 LangChain 등) → 0으로 안전 fallback.
    msg = _FakeMessage()
    assert extract_token_usage(msg) == (0, 0)


def test_extract_token_usage_empty_dict():
    # metadata 있지만 빈 dict — 토큰 키 누락 → 0.
    msg = _FakeMessage({})
    assert extract_token_usage(msg) == (0, 0)


# ---------------------------------------------------------------------------
# LLMCallMetrics.empty — 호출 생략 케이스 zero metrics
# ---------------------------------------------------------------------------


def test_llm_call_metrics_empty_factory():
    m = LLMCallMetrics.empty(step="content_gap", model="gpt-4o")
    assert m.step == "content_gap"
    assert m.model == "gpt-4o"
    assert m.cost_usd == 0.0
    assert m.latency_sec == 0.0
    assert m.prompt_tokens == 0
    assert m.completion_tokens == 0


# ---------------------------------------------------------------------------
# 실 OpenAI 호출 (skip-by-default)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not INTEGRATION_ENABLED, reason="VIDOCTOR_RUN_INTEGRATION=1 필요")
async def test_actual_call_traces_to_langfuse():
    """실 OpenAI 호출 + Langfuse 대시보드 trace 등록 확인용 (수동 검증)."""
    model = get_chat_model(model="gpt-4o-mini")
    response = await model.ainvoke("한 단어로만 답하라: 한국의 수도는?")
    assert "서울" in response.content
