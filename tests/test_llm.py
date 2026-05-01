"""LLM 래퍼 테스트.

기본 단위 테스트는 항상 실행 (모델 인스턴스 검증).
실 OpenAI 호출 + Langfuse trace 검증은 VIDOCTOR_RUN_INTEGRATION=1로 활성.
"""

import os

import pytest
from langchain_openai import ChatOpenAI

from vidoctor.llm import get_chat_model

INTEGRATION_ENABLED = os.environ.get("VIDOCTOR_RUN_INTEGRATION") == "1"


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
    callbacks = model.callbacks
    # callbacks는 list 또는 BaseCallbackManager. None이 아니면 부착됨
    assert callbacks is not None


@pytest.mark.skipif(not INTEGRATION_ENABLED, reason="VIDOCTOR_RUN_INTEGRATION=1 필요")
async def test_actual_call_traces_to_langfuse():
    """실 OpenAI 호출 + Langfuse 대시보드 trace 등록 확인용 (수동 검증).

    실행 후 Langfuse cloud 대시보드에서 'Vidoctor smoke test' trace가 보이는지 확인.
    """
    model = get_chat_model(model="gpt-4o-mini")
    response = await model.ainvoke("한 단어로만 답하라: 한국의 수도는?")
    assert "서울" in response.content
