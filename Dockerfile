# syntax=docker/dockerfile:1.7

# ---------- builder ----------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# 일부 transitive(grpcio 등)가 sdist 폴백 시 native 컴파일을 요구. 안전 마진.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv 공식 standalone 바이너리 — pip 우회로 빌드 속도·재현성↑
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# 의존성 캐시 레이어. lock·pyproject만 먼저 복사해 소스 변경 시 deps 재설치 회피.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 소스 복사 후 프로젝트 자체 install. --no-editable로 wheel 설치해 runtime stage의
# /app/.venv 복사만으로 모듈 import가 동작하게(editable은 source path 의존).
COPY src ./src
COPY README.md ./
RUN uv sync --frozen --no-dev --no-editable

# ---------- runtime ----------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# 분석은 Modal로 위임 — main 컨테이너는 yt-dlp(ffmpeg)와 멀티파트 업로드만 한다.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r app && useradd -r -g app -d /app -s /sbin/nologin app

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv ./.venv

USER app

EXPOSE 8000

CMD ["uvicorn", "vidoctor.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
