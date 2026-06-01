# syntax=docker/dockerfile:1.7

# ---------- builder ----------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# build-essential: 일부 의존성(silero-vad·whisperx transitive)이 native 컴파일을 요구.
# curl·ca-certificates는 uv를 COPY --from으로 받아 불필요.
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

# 런타임 시스템 의존성:
#   ffmpeg              — yt-dlp · whisperx 오디오 디코딩
#   libsndfile1         — librosa
#   libgl1, libglib2.0-0 — opencv 헤드리스 런타임
#   libgomp1            — torch OpenMP
#   libegl1, libgles2   — mediapipe BlazeFace/FaceLandmarker OpenGL ES 의존
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 libgl1 libglib2.0-0 libgomp1 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r app && useradd -r -g app -d /app -s /sbin/nologin app

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv ./.venv
COPY --chown=app:app src ./src
# KsponSpeech fine-tuned 모델(~1.5GB) + BlazeFace · FaceLandmarker 포함.
# 빌드 시점에 박아 cold start 시 외부 다운로드 의존을 없앤다.
COPY --chown=app:app models ./models

# 모델 캐시를 컨테이너 이미지에 통합 — worker subprocess가 매번 시작할 때 Pyannote
# 다운로드·Lightning checkpoint v1→v2 migration·ctranslate2 초기화를 다시 하지 않게.
# 더미 secrets는 Pydantic Settings 검증 통과용 (모델 로딩은 외부 API 호출 안 함).
ENV HF_HOME=/app/.cache/huggingface \
    XDG_CACHE_HOME=/app/.cache \
    MPLCONFIGDIR=/app/.cache/matplotlib \
    HOME=/app
RUN mkdir -p /app/.cache && \
    OPENAI_API_KEY=build-dummy \
    SUPABASE_URL=https://dummy.supabase.co \
    SUPABASE_SERVICE_KEY=build-dummy \
    R2_ENDPOINT=https://dummy.r2.com \
    R2_ACCESS_KEY_ID=build-dummy \
    R2_SECRET_ACCESS_KEY=build-dummy \
    R2_BUCKET=build-dummy \
    LANGFUSE_PUBLIC_KEY=build-dummy \
    LANGFUSE_SECRET_KEY=build-dummy \
    LANGFUSE_HOST=https://dummy.langfuse.com \
    VIDOCTOR_WHISPER_MODEL=/app/models/whisper-ko-ksponspeech-ct2 \
    /app/.venv/bin/python -c "from vidoctor.audio.transcribe import _load_models; _load_models()" && \
    chown -R app:app /app/.cache /app/models

USER app

EXPOSE 8000

# --workers 1: WhisperX·MediaPipe 모델이 worker마다 메모리 따로 잡혀 multi-worker는
# 4GB RAM 환경에서 OOM 위험. 동시성은 app.py 세마포어가 별도 제어.
CMD ["uvicorn", "vidoctor.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
