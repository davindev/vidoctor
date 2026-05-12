"""사용자 노출이 안전한 예외 베이스.

SSE error 이벤트 / FastAPI 응답에서 `str(exception)`을 그대로 노출하면 Supabase 내부
메시지·OpenAI 원본 오류·스택트레이스 단편이 새어나갈 위험이 있다. `SafeError`를 던지면
호출자(`api/app.py`)가 `public_message`만 사용자에게 보낸다 — 그 외 일반 예외는 generic
한국어 메시지로 일반화.

원인 예외 보존은 표준 `raise SafeError(...) from original` 사용.
"""

from __future__ import annotations


class SafeError(Exception):
    """user-facing 메시지를 명시한 예외. `public_message`는 사용자에게 그대로 노출 가능."""

    def __init__(self, public_message: str):
        super().__init__(public_message)
        self.public_message = public_message
