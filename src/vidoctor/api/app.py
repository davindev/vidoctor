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

import modal
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from vidoctor.api.youtube import YouTubeIngestError, download_youtube
from vidoctor.config import get_settings
from vidoctor.errors import SafeError
from vidoctor.graph import Category
from vidoctor.graph.state import (
    CATEGORY_DIMENSIONS,
    DIM_TO_STATE_FIELD,
    AnalysisState,
    ContentGapEvent,
    CPSEvent,
    DeadZoneEvent,
    FillerEvent,
    GazeEvent,
    Suggestion,
    Word,
)
from vidoctor.llm import LLMCallMetrics
from vidoctor.log_setup import analysis_id_var, configure_logging
from vidoctor.repository import (
    complete_analysis,
    count_analyses_today,
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

# worker 모델 로드(30~60초) + 분석(~5분) 여유분. hang 케이스만 끊고 정상은 통과.
_ANALYSIS_TIMEOUT_SEC = 15 * 60

# Modal이 분석 자체는 자동 스케일하지만 main의 R2 업로드·DB I/O 경합 방지 위해 5로 제한.
# 한도 초과 시 EventSource 단일 핸들러 일관성 위해 HTTP 상태 분기 대신 SSE error 이벤트로.
_MAX_CONCURRENT_ANALYSES = 5
_analysis_slot = asyncio.Semaphore(_MAX_CONCURRENT_ANALYSES)

# Modal에 배포된 분석 함수 — vidoctor_modal.py의 analyze_video.
_modal_analyze = modal.Function.from_name("vidoctor-analyze", "analyze_video")


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # 분석은 worker subprocess가 자체 로딩하므로 main에서 사전 로드 불필요.
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


def _client_ip(request: Request) -> str:
    """클라이언트 IP 추출. Fly proxy는 `fly-client-ip` 헤더에 실제 IP를 박고,
    일반 환경에서는 X-Forwarded-For 첫 항목을 사용. 둘 다 없으면 직접 연결 IP.
    """
    fly_ip = request.headers.get("fly-client-ip")
    if fly_ip:
        return fly_ip
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


limiter = Limiter(key_func=_client_ip)
app.state.limiter = limiter
# slowapi 핸들러 시그니처가 FastAPI 타입과 미스매치 — 공식 README 권장 패턴.
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]


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
async def list_recent_analyses(
    limit: int = Query(default=20, ge=1, le=100),
) -> list[AnalysisListItem]:
    """최근 분석 리스트를 반환한다."""
    rows = await asyncio.to_thread(list_analyses, limit)
    items: list[AnalysisListItem] = []
    for row in rows:
        video = row.get("videos") or {}
        items.append(
            AnalysisListItem(
                id=row["id"],
                started_at=row.get("started_at"),
                finished_at=row.get("finished_at"),
                error=row.get("error"),
                category=video.get("category"),
                storage_path=video.get("storage_path"),
                status=video.get("status"),
            )
        )
    return items


@app.get("/api/analyses/{analysis_id}", response_model=AnalysisDetail)
async def get_analysis(analysis_id: str) -> AnalysisDetail:
    """분석 단건 상세(meta + findings + suggestions)를 반환한다."""
    # 4개 쿼리 모두 독립적이라 gather로 병렬 fetch (영상당 ~0.4-1s 절감).
    meta, findings_raw, suggestions_raw, video_meta = await asyncio.gather(
        asyncio.to_thread(get_analysis_meta, analysis_id),
        asyncio.to_thread(get_analysis_findings, analysis_id),
        asyncio.to_thread(get_analysis_suggestions, analysis_id),
        asyncio.to_thread(get_analysis_video_meta, analysis_id),
    )
    if not meta:
        raise HTTPException(status_code=404, detail="분석을 찾을 수 없습니다.")

    findings: dict[str, list[FindingItem]] = {dim: [] for dim in DIM_TO_STATE_FIELD}
    for dimension, events in findings_raw.items():
        for event in events:
            data = event.model_dump()
            payload = {k: v for k, v in data.items() if k not in ("start", "end")}
            findings[dimension].append(
                FindingItem(
                    dimension=dimension,
                    start=float(data["start"]),
                    end=float(data["end"]),
                    payload=payload,
                )
            )

    suggestions = [
        SuggestionItem(text=s.text, finding_refs=s.finding_refs)
        for s in suggestions_raw
    ]

    metadata = cast(dict[str, Any], meta.get("metadata") or {})
    step_metrics = [StepMetric(**step) for step in metadata.get("step_metrics", [])]
    diarization_raw = metadata.get("speaker_diarization")
    diarization = (
        SpeakerDiarization(**diarization_raw) if diarization_raw else None
    )

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
        speaker_diarization=diarization,
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


def _deserialize_state(
    state_dict: dict[str, Any], video_path: str, category: Category
) -> AnalysisState:
    # worker._serialize_state와 대칭. complete_analysis가 model_dump를 호출하므로
    # 받은 dict를 Pydantic·dataclass 객체로 복원해야 함.
    return cast(
        AnalysisState,
        {
            "video_path": video_path,
            "category": category,
            "transcript": [Word(**w) for w in state_dict.get("transcript", [])],
            "fillers": [FillerEvent(**e) for e in state_dict.get("fillers", [])],
            "cps_anomalies": [
                CPSEvent(**e) for e in state_dict.get("cps_anomalies", [])
            ],
            "dead_zones": [
                DeadZoneEvent(**e) for e in state_dict.get("dead_zones", [])
            ],
            "gaze_issues": [GazeEvent(**e) for e in state_dict.get("gaze_issues", [])],
            "content_gaps": [
                ContentGapEvent(**e) for e in state_dict.get("content_gaps", [])
            ],
            "suggestions": [
                Suggestion(**s) for s in state_dict.get("suggestions", [])
            ],
            "step_metrics": [
                LLMCallMetrics(**m) for m in state_dict.get("step_metrics", [])
            ],
        },
    )


async def _save_upload_to_tmp(upload: UploadFile) -> tuple[Path, str]:
    """UploadFile을 청크 단위로 임시 파일에 떨궈 메모리 폭발 회피."""
    suffix = Path(upload.filename or "video.mp4").suffix or ".mp4"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        # 8MB 청크 — 큰 영상도 메모리 폭주 없이 스트리밍 저장.
        while chunk := await upload.read(8 * 1024 * 1024):
            tmp.write(chunk)
        return Path(tmp.name), (upload.filename or "video.mp4")


async def _analyze_stream(
    *,
    upload: UploadFile | None,
    url: str | None,
    category: Category | Literal["auto"],
) -> AsyncIterator[str]:
    """영상 입력 → R2 업로드 → worker subprocess 분석 → DB 저장. 진행은 SSE로 stream."""
    # 전체 사용자 합산 일일 한도 — IP rate limit이 우회되어도 비용 절대 상한 역할.
    today_count = await asyncio.to_thread(count_analyses_today)
    quota = get_settings().daily_quota
    if today_count >= quota:
        yield _sse(
            "error",
            {
                "message": (
                    f"오늘 분석 한도({quota}건)에 도달했습니다. 내일 다시 시도해주세요."
                ),
                "analysis_id": None,
            },
        )
        return

    # non-blocking acquire. 내부 수치 노출 회피 위해 메시지 일반화.
    try:
        async with asyncio.timeout(0):
            await _analysis_slot.acquire()
    except TimeoutError:
        yield _sse(
            "error",
            {
                "message": (
                    "현재 다른 분석이 진행 중입니다. 잠시 후 다시 시도해주세요."
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
                # endpoint XOR 검증이 빠졌을 때만 닿는 invariant 위반.
                raise RuntimeError("upload·url XOR invariant 위반")
            tmp_path, filename = await _save_upload_to_tmp(upload)

        # auto 분류와 R2 업로드는 입력 의존성이 없어 병렬 — classify가 cv2로 keyframe만
        # 읽기 때문에 동시 file read에 race 없음.
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

        # client가 graph 완료 전부터 결과 페이지를 polling할 수 있게 row를 먼저 만든다.
        video_id = await asyncio.to_thread(insert_video, storage_path, category, None)
        analysis_id = await asyncio.to_thread(insert_analysis, video_id)
        analysis_id_token = analysis_id_var.set(analysis_id)
        _log.info(
            "분석 시작",
            # LogRecord 표준 attr `filename`과 충돌 회피 위해 `video_filename` 사용.
            extra={"category": category, "video_filename": filename, "source": "url" if url else "file"},
        )
        yield _sse("started", {"analysis_id": analysis_id})
        yield _sse("uploaded", {})

        # Modal에 분석 위임. 컨테이너 격리 + auto-scale로 동시 N건 처리.
        # generator 소비를 별도 task로 두고 main은 15초마다 ping을 발송해 fetch idle 끊김 회피.
        modal_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        async def _consume_modal() -> None:
            try:
                async for event in _modal_analyze.remote_gen.aio(  # type: ignore[attr-defined]
                    storage_path, category, analysis_id
                ):
                    await modal_queue.put(("event", event))
            except Exception as exc:  # noqa: BLE001
                await modal_queue.put(("exception", exc))
            finally:
                await modal_queue.put(("done", None))

        consume_task = asyncio.create_task(_consume_modal())
        final_state_dict: dict[str, Any] | None = None
        worker_error: str | None = None

        try:
            async with asyncio.timeout(_ANALYSIS_TIMEOUT_SEC):
                while True:
                    try:
                        kind, payload = await asyncio.wait_for(
                            modal_queue.get(), timeout=15.0
                        )
                    except TimeoutError:
                        yield _sse("ping", {})
                        continue
                    if kind == "done":
                        break
                    if kind == "exception":
                        raise payload
                    event_type = payload.get("event")
                    if event_type == "node":
                        _log.info("graph 노드 완료", extra={"node": payload["name"]})
                        yield _sse("node", {"name": payload["name"]})
                    elif event_type == "complete":
                        final_state_dict = payload["state"]
                    elif event_type == "error":
                        worker_error = (
                            payload.get("message") or "분석 중 오류가 발생했습니다."
                        )
                    else:
                        _log.warning("알 수 없는 Modal 이벤트", extra={"event": event_type})
        finally:
            if not consume_task.done():
                consume_task.cancel()
                with suppress(Exception, asyncio.CancelledError, TimeoutError):
                    await asyncio.wait_for(consume_task, 5.0)

        if worker_error is not None:
            raise SafeError(worker_error)
        if final_state_dict is None:
            _log.error("Modal complete 이벤트 누락")
            raise SafeError("분석 중 오류가 발생했습니다.")
        graph_state = _deserialize_state(final_state_dict, str(tmp_path), category)

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
@limiter.limit("5/hour;15/day")
async def analyze(
    request: Request,
    category: str = Form(...),
    file: UploadFile | None = File(default=None),
    url: str | None = Form(default=None),
) -> StreamingResponse:
    """영상 업로드 또는 YouTube URL을 받아 분석을 시작한다 (SSE 진행 스트림).

    rate limit 한도 초과 시 slowapi가 429로 자동 응답 — 정확한 한도는 decorator 참고.
    """
    # graph 분기 안전성 위해 CATEGORY_DIMENSIONS 멤버 또는 "auto"만 통과.
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

    typed_category: Category | Literal["auto"] = (
        "auto" if category == "auto" else cast(Category, category)
    )
    return StreamingResponse(
        _analyze_stream(
            upload=file if has_file else None,
            url=url.strip() if has_url and url else None,
            category=typed_category,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
