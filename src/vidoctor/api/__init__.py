"""Vidoctor HTTP API — Next.js 프론트엔드용 FastAPI 진입점.

transport 계층. 비즈니스 로직은 graph/repository/suggestions에 위임.
"""

from vidoctor.api.app import app

__all__ = ["app"]
