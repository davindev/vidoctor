"""OpenAI LLM 래퍼 + Langfuse trace 통합.

- LangChain ChatOpenAI에 Langfuse callback을 부착해 모든 호출이 자동 trace.
- detect_content_gap (GPT-4o Vision), generate_suggestions (GPT-4o-mini)에서 공통 사용.
- Langfuse v4부터 글로벌 클라이언트 초기화 후 CallbackHandler가 그 상태를 공유하는 구조.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from vidoctor.config import get_settings


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
