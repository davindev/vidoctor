"""Vidoctor HTTP API — Next.js 프론트엔드용 FastAPI backbone.

기존 Streamlit UI(`vidoctor.ui.app`)의 비즈니스 로직을 그대로 재사용 (graph,
repository, build_suggestions 등). 이 모듈은 transport 계층만 담당.
"""

from vidoctor.api.app import app

__all__ = ["app"]
