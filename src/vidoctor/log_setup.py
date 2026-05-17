"""JSON 라인 logger + analysis_id correlation 컨텍스트.

`/api/analyze`가 분석 시작 시 `analysis_id_var.set(...)` 하면 같은 task 내 모든
로그 레코드에 `analysis_id` 자동 부착 — `jq '.analysis_id == "..."'`로 한 분석
로그 추출 가능.
"""

import json
import logging
import sys
from contextvars import ContextVar

# 한 분석의 ID. analyze 엔드포인트가 set, task 종료 시 ContextVar가 자동 격리.
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
    # uvicorn logger는 root로 통합 (자체 handler 없이 propagate).
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
