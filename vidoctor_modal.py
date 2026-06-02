"""Modal에 배포되는 분석 함수 — 동시 N건을 컨테이너 격리로 처리.

main(Fly)이 R2에 영상 업로드 후 storage_path를 이 함수에 넘기면, Modal이 자동
스케일된 컨테이너에서 R2 다운로드 → run_analysis → 진행 이벤트 stream → 결과 반환.
컨테이너 종료 시 OS가 모든 메모리를 회수해 누적 OOM 발생 안 함.

배포:
    modal deploy vidoctor_modal.py

호출 (main 측):
    import modal
    fn = modal.Function.from_name("vidoctor-analyze", "analyze_video")
    async for event in fn.remote_gen.aio(storage_path, category, analysis_id):
        ...
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
    cpu=8.0,
    memory=16384,
    timeout=900,
    # idle 5분 후 컨테이너 종료 — 메모리 회수 + 비용 절감.
    scaledown_window=300,
    # 동시 피크에 대비한 컨테이너 상한.
    max_containers=20,
)
async def analyze_video(
    storage_path: str,
    category: str,
    analysis_id: str,
):
    """R2 영상을 다운로드해 5차원 분석. 진행은 generator yield로 main에 stream.

    yield 형식 (1건당):
        {"event": "node", "name": "transcribe"}
        ...
        {"event": "complete", "state": {...직렬화된 결과...}}
        {"event": "error", "message": "사용자 노출 가능한 메시지"}
    """
    import asyncio
    import dataclasses
    import tempfile
    from pathlib import Path

    import boto3

    from vidoctor.config import get_settings
    from vidoctor.errors import SafeError
    from vidoctor.graph import run_analysis
    from vidoctor.log_setup import analysis_id_var, configure_logging

    configure_logging()
    analysis_id_var.set(analysis_id)

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
    try:
        await asyncio.to_thread(
            client.download_file, settings.r2_bucket, storage_path, str(tmp_path)
        )

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _on_node(name: str) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, {"event": "node", "name": name})

        async def _drive() -> None:
            try:
                state = await run_analysis(
                    str(tmp_path), category, on_node_complete=_on_node  # type: ignore[arg-type]
                )
                serialized = _serialize_state(dict(state))
                await queue.put({"event": "complete", "state": serialized})
            except Exception as e:  # noqa: BLE001
                public = (
                    e.public_message
                    if isinstance(e, SafeError)
                    else "분석 중 오류가 발생했습니다."
                )
                await queue.put({"event": "error", "message": public})

        drive_task = asyncio.create_task(_drive())
        try:
            while True:
                event = await queue.get()
                yield event
                if event["event"] in ("complete", "error"):
                    break
        finally:
            if not drive_task.done():
                drive_task.cancel()
    finally:
        tmp_path.unlink(missing_ok=True)


def _serialize_state(state: dict) -> dict:
    # audio_16k(ndarray)는 main에서 안 쓰고 직렬화도 무거워 제외.
    import dataclasses

    out: dict = {}
    for key, value in state.items():
        if key == "audio_16k":
            continue
        if isinstance(value, list):
            serialized: list = []
            for item in value:
                if hasattr(item, "model_dump"):
                    serialized.append(item.model_dump())
                elif dataclasses.is_dataclass(item) and not isinstance(item, type):
                    serialized.append(dataclasses.asdict(item))
                else:
                    serialized.append(item)
            out[key] = serialized
        else:
            out[key] = value
    return out
