"""FastAPI app — Next.js 프론트엔드용 HTTP 엔드포인트.

분석 CRUD + SSE 진행 스트림. 엔드포인트 상세는 /docs (OpenAPI) 참고.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal, cast

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from vidoctor.api.youtube import YouTubeIngestError, download_youtube
from vidoctor.config import get_settings
from vidoctor.errors import SafeError
from vidoctor.graph import Category, run_analysis
from vidoctor.graph.state import (
    CATEGORY_DIMENSIONS,
    DIM_TO_STATE_FIELD,
    AnalysisState,
)
from vidoctor.llm import LLMCallMetrics
from vidoctor.log_setup import analysis_id_var, configure_logging
from vidoctor.repository import (
    complete_analysis,
    create_video_signed_url,
    delete_video_for_analysis,
    fail_analysis,
    get_analysis_findings,
    get_analysis_meta,
    get_analysis_storage_path,
    get_analysis_suggestions,
    get_analysis_video_meta,
    insert_analysis,
    insert_video,
    list_analyses,
    upload_video_file,
)
from vidoctor.vision.category_classifier import classify_category

_log = logging.getLogger(__name__)

# 한 분석의 하드 cap. 10분 영상이라도 WhisperX/MediaPipe/GPT-4o Vision 합쳐 보통 ~5분.
# 15분이면 hang 케이스만 끊고 정상은 통과.
_ANALYSIS_TIMEOUT_SEC = 15 * 60

# 동시 진행 분석 cap. WhisperX·MediaPipe 모델 각 ~1-2GB RAM이라 무제한 동시는 OOM 위험.
# 시연 환경 단일 인스턴스 기준 2개. 한도 초과 시 SSE 채널을 연 뒤 첫 이벤트로 error를 보낸다
# (클라이언트가 EventSource 한 가지 핸들러로만 처리하도록 응답 코드 분기를 피함).
_MAX_CONCURRENT_ANALYSES = 2
_analysis_slot = asyncio.Semaphore(_MAX_CONCURRENT_ANALYSES)


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """프로세스 시작 시 JSON 로거 1회 구성. import side-effect로 두면 pytest caplog와
    충돌하므로 lifespan에서 명시적 호출."""
    configure_logging()
    yield


app = FastAPI(title="Vidoctor API", version="0.1.0", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(get_settings().frontend_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Fly.io / 외부 모니터의 readiness probe. 가벼운 응답만 — DB·외부 API는 점검 안 함."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class AnalysisListItem(BaseModel):
    """사이드바 분석 리스트 한 행 (videos JOIN 메타 포함)."""

    id: str
    started_at: str | None
    finished_at: str | None
    error: str | None
    category: str | None
    storage_path: str | None
    status: str | None


class FindingItem(BaseModel):
    """차원별 발견 사항 — start/end + 차원 고유 payload(JSONB)."""

    dimension: str
    start: float
    end: float
    payload: dict[str, Any]


class SuggestionItem(BaseModel):
    """LLM 개선 제안 + 참조 finding ref 리스트."""

    text: str
    finding_refs: list[str]


class StepMetric(BaseModel):
    """LLM 단계별 비용·latency·token 메타."""

    step: str
    model: str
    cost_usd: float
    latency_sec: float
    prompt_tokens: int
    completion_tokens: int


class SpeakerTurn(BaseModel):
    """화자 분리 단위 발화 구간 (start~end + 화자 식별자 + 텍스트 미리보기)."""

    start: float
    end: float
    speaker: str
    word_count: int
    text_preview: str


class SpeakerDiarization(BaseModel):
    """화자 분리 결과 — 주 화자 + 화자별 누적 시간 + turn 리스트."""

    main_speaker: str
    durations: dict[str, float]
    turns: list[SpeakerTurn]


class AnalysisDetail(BaseModel):
    """분석 상세 페이지 응답 — meta + findings + suggestions + step metrics."""

    id: str
    started_at: str | None
    finished_at: str | None
    cost_usd: float | None
    category: str | None
    storage_path: str | None
    duration_sec: float | None
    findings: dict[str, list[FindingItem]]
    suggestions: list[SuggestionItem]
    step_metrics: list[StepMetric]
    speaker_diarization: SpeakerDiarization | None


class VideoUrlResponse(BaseModel):
    """영상 R2 signed URL 응답 (영상 없으면 url=None)."""

    url: str | None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/analyses", response_model=list[AnalysisListItem])
async def list_recent_analyses(limit: int = 20) -> list[AnalysisListItem]:
    """최근 분석 리스트를 반환한다."""
    rows = await asyncio.to_thread(list_analyses, limit)
    items: list[AnalysisListItem] = []
    for r in rows:
        video = r.get("videos") or {}
        items.append(
            AnalysisListItem(
                id=r["id"],
                started_at=r.get("started_at"),
                finished_at=r.get("finished_at"),
                error=r.get("error"),
                category=video.get("category"),
                storage_path=video.get("storage_path"),
                status=video.get("status"),
            )
        )
    return items


@app.get("/api/analyses/{analysis_id}", response_model=AnalysisDetail)
async def get_analysis(analysis_id: str) -> AnalysisDetail:
    """분석 단건 상세(meta + findings + suggestions)를 반환한다."""
    # 4개 쿼리가 모두 analysis_id만 받고 서로 의존 없음 → gather로 병렬 fetch.
    # 순차였을 때 0.4~1.2s, 병렬은 최장 호출 시간 1개로 수렴.
    meta, findings_raw, suggestions_raw, video_meta = await asyncio.gather(
        asyncio.to_thread(get_analysis_meta, analysis_id),
        asyncio.to_thread(get_analysis_findings, analysis_id),
        asyncio.to_thread(get_analysis_suggestions, analysis_id),
        asyncio.to_thread(get_analysis_video_meta, analysis_id),
    )
    if not meta:
        raise HTTPException(status_code=404, detail="분석을 찾을 수 없습니다.")

    findings: dict[str, list[FindingItem]] = {dim: [] for dim in DIM_TO_STATE_FIELD}
    for dim, events in findings_raw.items():
        for ev in events:
            data = ev.model_dump()
            payload = {k: v for k, v in data.items() if k not in ("start", "end")}
            findings[dim].append(
                FindingItem(
                    dimension=dim,
                    start=float(data["start"]),
                    end=float(data["end"]),
                    payload=payload,
                )
            )

    suggestions = [
        SuggestionItem(text=s.text, finding_refs=s.finding_refs) for s in suggestions_raw
    ]

    metadata = cast(dict[str, Any], meta.get("metadata") or {})
    step_metrics = [StepMetric(**sm) for sm in metadata.get("step_metrics", [])]
    diar_raw = metadata.get("speaker_diarization")
    diar = SpeakerDiarization(**diar_raw) if diar_raw else None

    return AnalysisDetail(
        id=analysis_id,
        started_at=meta.get("started_at"),
        finished_at=meta.get("finished_at"),
        cost_usd=meta.get("cost_usd"),
        category=video_meta.get("category") if video_meta else None,
        storage_path=video_meta.get("storage_path") if video_meta else None,
        duration_sec=video_meta.get("duration_sec") if video_meta else None,
        findings=findings,
        suggestions=suggestions,
        step_metrics=step_metrics,
        speaker_diarization=diar,
    )


@app.get("/api/analyses/{analysis_id}/video-url", response_model=VideoUrlResponse)
async def get_video_url(analysis_id: str) -> VideoUrlResponse:
    """영상 R2 객체의 signed URL(2시간)을 발급한다."""
    storage_path = await asyncio.to_thread(get_analysis_storage_path, analysis_id)
    if storage_path is None:
        return VideoUrlResponse(url=None)
    url = await asyncio.to_thread(create_video_signed_url, storage_path, 7200)
    return VideoUrlResponse(url=url)


@app.delete("/api/analyses/{analysis_id}")
async def delete_analysis(analysis_id: str) -> dict[str, str]:
    """영상 + 모든 관련 분석·findings·suggestions를 삭제한다."""
    try:
        await asyncio.to_thread(delete_video_for_analysis, analysis_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# POST /api/analyze
# ---------------------------------------------------------------------------


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _save_upload_to_tmp(uploaded: UploadFile) -> tuple[Path, str]:
    """UploadFile을 청크 단위로 임시 파일에 떨궈 메모리 폭발 회피."""
    suffix = Path(uploaded.filename or "video.mp4").suffix or ".mp4"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        # 8MB 청크 — 큰 영상도 메모리 폭주 없이 스트리밍 저장.
        while chunk := await uploaded.read(8 * 1024 * 1024):
            tmp.write(chunk)
        return Path(tmp.name), (uploaded.filename or "video.mp4")


async def _analyze_stream(
    *,
    upload: UploadFile | None,
    url: str | None,
    category: Category | Literal["auto"],
) -> AsyncIterator[str]:
    """영상(파일 업로드 또는 유튜브 URL)을 임시 파일에 떨군 뒤 R2 업로드 → graph 실행 → DB 저장.

    각 단계 시점에 SSE 이벤트를 yield. 클라이언트는 `EventSource` 또는 fetch+ReadableStream
    으로 구독해 진행 그래프를 갱신.

    `_analysis_slot` 세마포어로 동시 분석 수를 제한 — 모델 메모리(~1-2GB × N)가 OOM
    유발하지 않도록. 큐가 꽉 차면 즉시 사용자에게 안내 메시지를 보내고 종료.
    """
    # 슬롯이 모두 차 있으면 즉시 TimeoutError → SSE error 이벤트로 안내.
    try:
        async with asyncio.timeout(0):
            await _analysis_slot.acquire()
    except TimeoutError:
        yield _sse(
            "error",
            {
                "message": (
                    f"동시 분석 {_MAX_CONCURRENT_ANALYSES}건 한도에 도달했습니다. "
                    "잠시 후 다시 시도해주세요."
                ),
                "analysis_id": None,
            },
        )
        return

    tmp_path: Path | None = None
    analysis_id: str | None = None
    video_id: str | None = None
    classify_metrics: LLMCallMetrics | None = None
    analysis_id_token = None

    async def _safe_fail(reason: str) -> None:
        """fail_analysis 베스트 에포트 — DB row가 in-progress로 영구 남지 않게."""
        if analysis_id and video_id:
            with suppress(Exception):
                await asyncio.to_thread(fail_analysis, analysis_id, video_id, reason)

    try:
        # 1) 소스 → 로컬 tmp.
        if url is not None:
            yield _sse("status", {"phase": "downloading"})
            try:
                tmp_path, title = await download_youtube(url)
            except YouTubeIngestError as e:
                yield _sse("error", {"message": str(e), "analysis_id": None})
                return
            filename = f"{title}.mp4"
            yield _sse("metadata", {"filename": filename})
        else:
            if upload is None:
                # endpoint XOR 검증이 빠졌을 때만 발생하는 invariant 위반.
                raise RuntimeError("upload·url XOR invariant 위반")
            tmp_path, filename = await _save_upload_to_tmp(upload)

        # 2) auto면 분류(~1-2s)와 R2 업로드(수 초)를 병렬 실행 — 둘 다 IO bound이고
        # 입력 의존성이 없다. classify는 cv2 POS_MSEC seek로 키프레임만 읽어 동시 접근에
        # race가 없으며, videos row 삽입 직전까지 둘 다 완료되면 된다.
        if category == "auto":
            yield _sse("status", {"phase": "classifying"})
            classify_task = asyncio.create_task(classify_category(str(tmp_path)))
            upload_task = asyncio.create_task(
                asyncio.to_thread(upload_video_file, tmp_path, filename)
            )
            category, classify_metrics = await classify_task
            yield _sse("category", {"category": category})
            yield _sse("status", {"phase": "uploading"})
            storage_path = await upload_task
        else:
            yield _sse("status", {"phase": "uploading"})
            storage_path = await asyncio.to_thread(upload_video_file, tmp_path, filename)

        # videos·analyses row 생성 후 analysis_id 통지 (graph 진행 중에도 client가 폴링 가능).
        video_id = await asyncio.to_thread(insert_video, storage_path, category, None)
        analysis_id = await asyncio.to_thread(insert_analysis, video_id)
        # 이 분석의 모든 로그에 analysis_id 자동 부착. Token은 finally에서 reset.
        analysis_id_token = analysis_id_var.set(analysis_id)
        _log.info(
            "분석 시작",
            # LogRecord 표준 attr `filename`(호출 소스 파일명)과 충돌하므로 `video_filename` 사용.
            extra={"category": category, "video_filename": filename, "source": "url" if url else "file"},
        )
        yield _sse("started", {"analysis_id": analysis_id})
        yield _sse("uploaded", {})

        # graph의 sync callback을 메인 loop으로 안전하게 옮기는 큐.
        # graph 종료 시 sentinel을 넣어 while-loop이 깔끔히 종료.
        sentinel = object()
        loop = asyncio.get_running_loop()
        node_queue: asyncio.Queue[Any] = asyncio.Queue()

        def _on_node(name: str) -> None:
            # sync 노드에서 콜백되므로 main loop으로 thread-safe하게 schedule (contextvar 전파).
            loop.call_soon_threadsafe(
                lambda: _log.info("graph 노드 완료", extra={"node": name})
            )
            loop.call_soon_threadsafe(
                node_queue.put_nowait, ("node", {"name": name})
            )

        async def _drive_graph() -> AnalysisState:
            try:
                return await run_analysis(
                    str(tmp_path), category, on_node_complete=_on_node
                )
            finally:
                node_queue.put_nowait(sentinel)

        graph_task = asyncio.create_task(_drive_graph())

        try:
            # 전체 분석 hard cap (상단 _ANALYSIS_TIMEOUT_SEC 정의 참고).
            async with asyncio.timeout(_ANALYSIS_TIMEOUT_SEC):
                while True:
                    item = await node_queue.get()
                    if item is sentinel:
                        break
                    event_name, payload = item
                    yield _sse(event_name, payload)
                graph_state = await graph_task
        finally:
            # 비정상 종료 시 graph_task 누수 방지. cancel 후 5초 대기, 그 이상은 포기
            # (sync 모델 호출은 cooperative cancel 불가).
            if not graph_task.done():
                graph_task.cancel()
                with suppress(Exception, asyncio.CancelledError, TimeoutError):
                    await asyncio.wait_for(graph_task, 5.0)

        # 분류기는 graph 바깥에서 실행되므로 메트릭을 수동으로 합산.
        if classify_metrics is not None:
            existing = graph_state.get("step_metrics") or []
            graph_state["step_metrics"] = [*existing, classify_metrics]

        await complete_analysis(analysis_id, video_id, graph_state)
        total_cost = sum(m.cost_usd for m in (graph_state.get("step_metrics") or []))
        _log.info("분석 완료", extra={"total_cost_usd": round(total_cost, 6)})
        yield _sse("complete", {"analysis_id": analysis_id})

    except (asyncio.CancelledError, GeneratorExit):
        # 클라이언트 disconnect — DB row가 in-progress로 영구히 남지 않게 fail 처리.
        await _safe_fail("클라이언트 연결 끊김")
        raise
    except TimeoutError:
        _log.warning("분석 파이프라인 타임아웃")
        await _safe_fail("분석 타임아웃")
        yield _sse(
            "error",
            {
                "message": (
                    "분석 시간이 초과되어 중단되었습니다. "
                    "더 짧은 영상으로 다시 시도해주세요."
                ),
                "analysis_id": analysis_id,
            },
        )
    except Exception as e:  # noqa: BLE001
        _log.exception("분석 파이프라인 실패")
        await _safe_fail(str(e))
        # 내부 예외 메시지 그대로 노출하면 Supabase/OpenAI raw error가 새어나감.
        # SafeError처럼 의도적으로 user-facing인 예외만 메시지 그대로, 나머지는 일반화.
        public = e.public_message if isinstance(e, SafeError) else "분석 중 오류가 발생했습니다."
        yield _sse("error", {"message": public, "analysis_id": analysis_id})
    finally:
        if analysis_id_token is not None:
            analysis_id_var.reset(analysis_id_token)
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        _analysis_slot.release()


@app.post("/api/analyze")
async def analyze(
    category: str = Form(...),
    file: UploadFile | None = File(default=None),
    url: str | None = Form(default=None),
) -> StreamingResponse:
    """영상 업로드 또는 YouTube URL을 받아 분석을 시작한다 (SSE 진행 스트림)."""
    # "auto"는 분류기 위임. 그 외에는 CATEGORY_DIMENSIONS 멤버여야 graph 분기가 안전.
    if category != "auto" and category not in CATEGORY_DIMENSIONS:
        raise HTTPException(status_code=400, detail=f"알 수 없는 카테고리: {category}")
    has_file = file is not None and (file.filename or "") != ""
    has_url = url is not None and url.strip() != ""
    if has_file and has_url:
        raise HTTPException(
            status_code=400, detail="file과 url은 동시에 보낼 수 없습니다."
        )
    if not has_file and not has_url:
        raise HTTPException(
            status_code=400, detail="file 또는 url을 제공해야 합니다."
        )

    cat: Category | Literal["auto"]
    if category == "auto":
        cat = "auto"
    else:
        cat = cast(Category, category)
    return StreamingResponse(
        _analyze_stream(
            upload=file if has_file else None,
            url=url.strip() if has_url and url else None,
            category=cat,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
