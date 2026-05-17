"""사용자 노출이 안전한 예외 베이스.

`str(exception)` 그대로 노출하면 Supabase/OpenAI 내부 메시지·스택트레이스 단편이
새어나갈 위험. SafeError를 던지면 호출자가 `public_message`만 사용자에 노출,
나머지 일반 예외는 generic 메시지로 일반화한다.
"""


class SafeError(Exception):
    """user-facing 메시지를 명시한 예외. `public_message`는 사용자에게 그대로 노출 가능."""

    def __init__(self, public_message: str):
        super().__init__(public_message)
        self.public_message = public_message
