"""Modal에 배포되는 분석 함수 — 동시 N건을 컨테이너 격리로 처리.

main(Fly)이 R2에 영상 업로드 후 storage_path를 이 함수에 넘기면, Modal이 자동
스케일된 컨테이너에서 R2 다운로드 → run_analysis를 실행하고, 진행률·결과·실패를
DB(Supabase)에 직접 기록한다. main은 spawn 후 기다리지 않으므로 연결이 끊겨도
분석이 끝까지 보존된다. 컨테이너 종료 시 OS가 모든 메모리를 회수해 누적 OOM이 발생하지 않는다.

배포:
    modal deploy vidoctor_modal.py

호출 (main 측):
    import modal
    fn = modal.Function.from_name("vidoctor-analyze", "analyze_video")
    await fn.spawn.aio(storage_path, category, analysis_id, video_id, classify_metric)
"""

from __future__ import annotations

from pathlib import Path

import modal

# vidoctor 패키지 + 모델 + 시스템 의존성을 박은 이미지. 모델 로딩(lightning migration·
# Pyannote 캐시)을 빌드 시점에 1회 실행해 runtime 콜드 스타트를 최소화.
_ROOT = Path(__file__).parent

vidoctor_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ffmpeg",
        "libsndfile1",
        "libgl1",
        "libglib2.0-0",
        "libgomp1",
        "libegl1",
        "libgles2",
        "build-essential",
    )
    .pip_install("uv")
    # uv lock 기반 의존성 설치. Modal 이미지에 venv를 박아 빠른 import.
    .add_local_file(str(_ROOT / "pyproject.toml"), "/app/pyproject.toml", copy=True)
    .add_local_file(str(_ROOT / "uv.lock"), "/app/uv.lock", copy=True)
    .add_local_file(str(_ROOT / "README.md"), "/app/README.md", copy=True)
    .workdir("/app")
    .run_commands(
        # analyze extra로 ML 의존성(whisperx·mediapipe·torch CPU 계열·librosa·silero·
        # scenedetect·mlflow)까지 설치. fly Dockerfile은 extra 없이 base만.
        "uv sync --frozen --no-dev --no-install-project --link-mode=copy --extra analyze",
    )
    .env(
        {
            "PATH": "/app/.venv/bin:/usr/local/bin:/usr/bin:/bin",
            # ctranslate2(GPU)가 cuDNN·cuBLAS를 찾도록 — torch CUDA wheel이 설치한 nvidia
            # 라이브러리 경로. 미설정 시 libcudnn 로드 실패로 GPU transcribe가 죽는다.
            "LD_LIBRARY_PATH": (
                "/app/.venv/lib/python3.11/site-packages/nvidia/cublas/lib:"
                "/app/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib"
            ),
            "VIDOCTOR_WHISPER_MODEL": "/app/models/whisper-ko-ksponspeech-ct2",
            "HF_HOME": "/app/.cache/huggingface",
            "XDG_CACHE_HOME": "/app/.cache",
            "MPLCONFIGDIR": "/app/.cache/matplotlib",
            "HOME": "/app",
        }
    )
    .add_local_dir(str(_ROOT / "src"), "/app/src", copy=True)
    .add_local_dir(str(_ROOT / "models"), "/app/models", copy=True)
    .run_commands(
        # vidoctor 패키지 install + 모델 pre-warm (lightning migration + Pyannote 캐시).
        # 더미 secrets는 Pydantic Settings 검증 통과용. runtime엔 Modal secret이 덮어씀.
        "cd /app && uv pip install --no-deps -e . --link-mode=copy",
        "cd /app && OPENAI_API_KEY=dummy SUPABASE_URL=https://x.supabase.co "
        "SUPABASE_SERVICE_KEY=dummy R2_ENDPOINT=https://x.r2.com R2_ACCESS_KEY_ID=dummy "
        "R2_SECRET_ACCESS_KEY=dummy R2_BUCKET=dummy LANGFUSE_PUBLIC_KEY=dummy "
        "LANGFUSE_SECRET_KEY=dummy LANGFUSE_HOST=https://x.langfuse.com "
        "/app/.venv/bin/python -c 'from vidoctor.audio.transcribe import _load_models; _load_models()'",
    )
)

app = modal.App("vidoctor-analyze", image=vidoctor_image)

# Modal CLI로 사전 등록한 secret. 사용자가 `modal secret create vidoctor-secrets ...`로 생성.
vidoctor_secret = modal.Secret.from_name("vidoctor-secrets")


@app.function(
    secrets=[vidoctor_secret],
    # transcribe(WhisperX/ctranslate2)를 GPU로 — CPU 대비 전사 시간 단축.
    gpu="T4",
    cpu=8.0,
    memory=16384,
    timeout=900,
    # GPU는 idle 비용이 커서 짧게 — 직후 요청만 웜으로 흡수하고 반납.
    scaledown_window=120,
    # 동시 피크에 대비한 컨테이너 상한.
    max_containers=20,
)
async def analyze_video(
    storage_path: str,
    category: str,
    analysis_id: str,
    video_id: str | None = None,
    classify_metric: dict | None = None,
):
    """R2 영상을 다운로드해 5차원 분석 후 결과를 DB에 직접 저장한다.

    이전에는 결과를 generator로 main(Fly)에 yield해 Fly가 저장했으나, 클라이언트
    연결이 끊기면 분석이 통째로 유실됐다. 이제 Modal이 detached로 끝까지 실행하며
    진행률(update_progress)·결과(complete_analysis)·실패(fail_analysis)를 스스로
    기록한다. video_id·classify_metric은 배포 윈도우 호환을 위해 옵셔널이나 항상 전달된다.
    """
    import os

    # _load_models가 이 변수로 device를 정한다. 여기서 cuda를 박아 런타임 추론은 GPU로,
    # 빌드 prewarm 단계는 이 변수가 없어 CPU로 로드된다(prewarm은 모델 캐싱 목적이라 무방).
    os.environ.setdefault("VIDOCTOR_DEVICE", "cuda")

    import asyncio
    import logging
    import tempfile
    from contextlib import suppress
    from pathlib import Path
    from typing import cast

    import boto3
    from botocore.exceptions import ClientError

    from vidoctor.config import get_settings
    from vidoctor.errors import SafeError
    from vidoctor.graph import run_analysis
    from vidoctor.llm import LLMCallMetrics
    from vidoctor.log_setup import analysis_id_var, configure_logging
    from vidoctor.repository import complete_analysis, fail_analysis, update_progress

    configure_logging()
    analysis_id_var.set(analysis_id)
    log = logging.getLogger("vidoctor.modal")
    vid = cast(str, video_id)

    # R2 → 임시 파일.
    settings = get_settings()
    client = boto3.client(
        "s3",
        endpoint_url=settings.r2_endpoint,
        aws_access_key_id=settings.r2_access_key_id.get_secret_value(),
        aws_secret_access_key=settings.r2_secret_access_key.get_secret_value(),
        region_name="auto",
    )
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    def _on_node(name: str) -> None:
        # 진행률은 best-effort — 기록 실패가 분석을 막지 않는다.
        try:
            update_progress(analysis_id, name)
        except Exception:
            log.warning("progress 기록 실패", extra={"node": name})

    try:
        try:
            await asyncio.to_thread(
                client.download_file, settings.r2_bucket, storage_path, str(tmp_path)
            )
        except ClientError as e:
            # 404(키 없음)와 그 외(권한·일시 오류)를 구분해 사용자에게 정확히 안내.
            code = str(e.response.get("Error", {}).get("Code"))
            msg = (
                "영상 파일을 찾을 수 없습니다."
                if code in ("404", "NoSuchKey")
                else "영상을 불러오지 못했습니다."
            )
            raise SafeError(msg) from e
        # self-timeout(< Modal timeout 900s) — hang 시 스스로 fail 기록할 여지를 남긴다.
        async with asyncio.timeout(870):
            state = await run_analysis(
                str(tmp_path), category, on_node_complete=_on_node  # type: ignore[arg-type]
            )
        # 분류기 메트릭은 graph 바깥(Fly)에서 생성돼 인자로 넘어옴 — 수동 합산.
        if classify_metric is not None:
            existing = state.get("step_metrics") or []
            state["step_metrics"] = [*existing, LLMCallMetrics(**classify_metric)]
        await complete_analysis(analysis_id, vid, state)
        log.info("분석 완료")
    except TimeoutError:
        log.warning("분석 타임아웃")
        with suppress(Exception):
            await asyncio.to_thread(
                fail_analysis, analysis_id, vid, "분석 시간이 초과되었습니다."
            )
    except Exception as e:  # noqa: BLE001
        public = (
            e.public_message
            if isinstance(e, SafeError)
            else "분석 중 오류가 발생했습니다."
        )
        log.exception("분석 실패")
        with suppress(Exception):
            await asyncio.to_thread(fail_analysis, analysis_id, vid, public)
    finally:
        tmp_path.unlink(missing_ok=True)
