"""JSON 라인 logger + analysis_id correlation 컨텍스트.

`/api/analyze`가 한 분석 시작 시 `analysis_id_var.set(...)`로 컨텍스트를 주입하면,
이후 같은 task 내에서 발생하는 모든 로그 레코드에 `analysis_id`가 자동 부착된다.
운영 환경에서 `jq '.analysis_id == "..."'` 한 줄로 한 분석의 로그를 추출 가능.

표준 logging Formatter에 contextvar를 직접 끌어오는 방법이 깔끔. 별도 라이브러리(structlog
등) 도입은 prototype 범위 초과 — 필요 최소만 구현.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar

# 한 분석의 ID. analyze 엔드포인트 진입 시 set, exception/완료 후 해제는 ContextVar의
# task-local 격리로 자동.
analysis_id_var: ContextVar[str | None] = ContextVar("analysis_id", default=None)


# stdlib LogRecord 표준 attr — dummy 인스턴스에서 자동 derive. Python 패치 버전이 새
# attr을 추가해도 자동 추종된다. `asctime`/`message`/`taskName`은 포맷팅 중에만 들어가
# dict에 없으므로 명시 추가.
_RESERVED_LOGRECORD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"asctime", "message", "taskName"}


class _JsonFormatter(logging.Formatter):
    """레코드를 JSON 한 줄로 직렬화 + analysis_id contextvar 자동 부착."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        aid = analysis_id_var.get()
        if aid is not None:
            payload["analysis_id"] = aid
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # logger.info("...", extra={"key": ...})로 들어온 사용자 정의 필드 흡수.
        for key, value in record.__dict__.items():
            if key in _RESERVED_LOGRECORD_ATTRS or key.startswith("_") or key in payload:
                continue
            payload[key] = value
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """프로세스 진입 1회 호출. root logger를 JSON 라인으로 재구성하고 uvicorn loggers는
    핸들러 없이 propagate=True로 묶어 단일 출력 경로로 통합한다."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    # 기존 핸들러 제거 (uvicorn default가 평문 stderr라 둘이 섞이면 grep 망가짐).
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level)
    # uvicorn logger는 자체 handler 갖지 말고 root로 흘려보내 단일 출력 — stdlib cookbook
    # "library code attaches NullHandler; application code configures root" 패턴.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
