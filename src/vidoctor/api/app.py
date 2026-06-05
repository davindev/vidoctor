"""FastAPI app — Next.js 프론트엔드용 HTTP 엔드포인트.

분석 CRUD + 진행 상태 조회(폴링).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal, cast
from uuid import uuid4

import modal
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from vidoctor.api.youtube import YouTubeIngestError, download_youtube
from vidoctor.config import get_settings
from vidoctor.graph import Category
from vidoctor.graph.state import (
    CATEGORY_DIMENSIONS,
    DIM_TO_STATE_FIELD,
)
from vidoctor.llm import LLMCallMetrics
from vidoctor.log_setup import configure_logging
from vidoctor.repository import (
    STALE_THRESHOLD_SEC,
    count_analyses_today,
    count_in_progress_analyses,
    create_video_signed_url,
    delete_video_for_analysis,
    fail_analysis,
    get_analysis_findings,
    get_analysis_meta,
    get_analysis_status,
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

# 진행 중(status='analyzing') 분석 동시 상한. 분석은 Modal이 detached로 끝까지 돌고
# Fly는 기다리지 않으므로 Semaphore 대신 DB count로 가드한다. 정확한 원자 상한이 아니라
# 비용 폭증을 막는 soft cap이며, 하드 상한은 Modal max_containers가 담당한다.
_MAX_CONCURRENT_ANALYSES = 5

# Modal에 배포된 분석 함수 — vidoctor_modal.py의 analyze_video.
_modal_analyze = modal.Function.from_name("vidoctor-analyze", "analyze_video")


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # 분석은 Modal 컨테이너가 자체 로딩하므로 main에서 사전 로드 불필요.
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
    filename: str | None


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
    filename: str | None
    findings: dict[str, list[FindingItem]]
    suggestions: list[SuggestionItem]
    step_metrics: list[StepMetric]
    speaker_diarization: SpeakerDiarization | None


class VideoUrlResponse(BaseModel):
    """영상 R2 signed URL 응답 (영상 없으면 url=None)."""

    url: str | None


class AnalyzeResponse(BaseModel):
    """분석 시작 응답 — 이후 클라이언트는 이 id로 status를 폴링한다."""

    analysis_id: str


class AnalysisStatusResponse(BaseModel):
    """폴링용 진행 상태 — status(analyzing|completed|failed) + 노드 진행률 + 에러."""

    status: str | None
    progress: dict[str, Any]
    error: str | None


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
        status = video.get("status")
        # Modal 강제종료로 fail 기록조차 못한 stale row는 목록에서도 failed로 보여, 사이드바가
        # 영구 'analyzing'으로 남아 목록 폴링이 무한히 도는 것을 막는다(DB 정리는 status 조회 담당).
        if status == "analyzing" and _is_stale(row.get("started_at")):
            status = "failed"
        items.append(
            AnalysisListItem(
                id=row["id"],
                started_at=row.get("started_at"),
                finished_at=row.get("finished_at"),
                error=row.get("error"),
                category=video.get("category"),
                storage_path=video.get("storage_path"),
                status=status,
                filename=video.get("filename"),
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
        filename=video_meta.get("filename") if video_meta else None,
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


async def _save_upload_to_tmp(upload: UploadFile) -> tuple[Path, str]:
    """UploadFile을 청크 단위로 임시 파일에 떨궈 메모리 폭발 회피."""
    suffix = Path(upload.filename or "video.mp4").suffix or ".mp4"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        # 8MB 청크 — 큰 영상도 메모리 폭주 없이 스트리밍 저장.
        while chunk := await upload.read(8 * 1024 * 1024):
            tmp.write(chunk)
        return Path(tmp.name), (upload.filename or "video.mp4")


async def _run_analyze(
    *,
    upload: UploadFile | None,
    url: str | None,
    category: Category | Literal["auto"],
) -> str:
    """영상 입력 → R2 업로드 → Modal에 detached 위임 → analysis_id 반환.

    분석 자체는 Fly가 기다리지 않는다 — Modal이 끝까지 실행하며 진행률·결과를 DB에
    직접 저장하므로, 클라이언트 연결이 끊겨도 분석은 보존된다. 여기서는 다운로드·
    업로드·분류까지만 동기로 처리하고 spawn 직후 즉시 반환한다. 진행은 폴링으로 추적.
    """
    # 전체 사용자 합산 일일 한도 — IP rate limit이 우회되어도 비용 절대 상한 역할.
    today_count = await asyncio.to_thread(count_analyses_today)
    quota = get_settings().daily_quota
    if today_count >= quota:
        raise HTTPException(
            status_code=429,
            detail=f"오늘 분석 한도({quota}건)에 도달했습니다. 내일 다시 시도해주세요.",
        )

    # 진행 중 동시 분석 soft cap. 내부 수치 노출 회피 위해 메시지 일반화.
    in_progress = await asyncio.to_thread(count_in_progress_analyses)
    if in_progress >= _MAX_CONCURRENT_ANALYSES:
        raise HTTPException(
            status_code=409,
            detail="현재 다른 분석이 진행 중입니다. 잠시 후 다시 시도해주세요.",
        )

    tmp_path: Path | None = None
    try:
        if url is not None:
            try:
                tmp_path, title = await download_youtube(url)
            except YouTubeIngestError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            filename = f"{title}.mp4"
        else:
            if upload is None:
                # endpoint XOR 검증이 빠졌을 때만 닿는 invariant 위반.
                raise RuntimeError("upload·url XOR invariant 위반")
            tmp_path, filename = await _save_upload_to_tmp(upload)

        # R2 키는 파일명 특수문자(따옴표·슬래시 등)가 새지 않도록 uuid 기반 안전 키로
        # 고정하고, 원본 파일명은 videos.filename에 따로 보관한다.
        suffix = Path(filename).suffix or ".mp4"
        storage_key = f"videos/{uuid4().hex}{suffix}"
        # auto 분류와 R2 업로드는 입력 의존성이 없어 병렬 — classify가 cv2로 keyframe만
        # 읽기 때문에 동시 file read에 race 없음.
        classify_metrics: LLMCallMetrics | None = None
        if category == "auto":
            classify_task = asyncio.create_task(classify_category(str(tmp_path)))
            upload_task = asyncio.create_task(
                asyncio.to_thread(upload_video_file, tmp_path, storage_key)
            )
            category, classify_metrics = await classify_task
            storage_path = await upload_task
        else:
            storage_path = await asyncio.to_thread(upload_video_file, tmp_path, storage_key)

        video_id = await asyncio.to_thread(
            insert_video, storage_path, category, None, filename
        )
        analysis_id = await asyncio.to_thread(insert_analysis, video_id)
        _log.info(
            "분석 시작",
            extra={
                "category": category,
                # LogRecord 표준 attr `filename`과 충돌 회피 위해 `video_filename` 사용.
                "video_filename": filename,
                "source": "url" if url else "file",
                "analysis_id": analysis_id,
            },
        )

        # Modal에 detached 위임 — Fly 연결·재배포와 무관하게 끝까지 실행되며 결과를 DB에 저장.
        # LLMCallMetrics는 dataclass — Modal에 넘길 직렬화 가능한 dict로 변환.
        classify_metric = asdict(classify_metrics) if classify_metrics else None
        try:
            await _modal_analyze.spawn.aio(  # type: ignore[attr-defined]
                storage_path, category, analysis_id, video_id, classify_metric
            )
        except Exception as e:  # noqa: BLE001
            # spawn 실패 시 row가 영원히 analyzing으로 남지 않게 즉시 정리(20분 stale 대기 회피).
            _log.exception("Modal spawn 실패")
            with suppress(Exception):
                await asyncio.to_thread(
                    fail_analysis, analysis_id, video_id, "분석을 시작하지 못했습니다."
                )
            raise HTTPException(
                status_code=502,
                detail="분석을 시작하지 못했습니다. 잠시 후 다시 시도해주세요.",
            ) from e

        return analysis_id
    except HTTPException:
        # 위에서 의도적으로 던진 400/502 등은 그대로 전달.
        raise
    except Exception as e:  # noqa: BLE001
        # R2 업로드·DB insert 등 예상 못한 실패 — 내부 예외를 노출하지 않고 일반화.
        _log.exception("분석 시작 처리 실패")
        raise HTTPException(
            status_code=500,
            detail="분석을 시작하지 못했습니다. 잠시 후 다시 시도해주세요.",
        ) from e
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


@app.post("/api/analyze", response_model=AnalyzeResponse)
@limiter.limit("5/hour;15/day")
async def analyze(
    request: Request,
    category: str = Form(...),
    file: UploadFile | None = File(default=None),
    url: str | None = Form(default=None),
) -> AnalyzeResponse:
    """영상 업로드 또는 YouTube URL을 받아 분석을 시작하고 analysis_id를 반환한다.

    분석 진행은 GET /api/analyses/{id}/status 폴링으로 추적한다.
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
    analysis_id = await _run_analyze(
        upload=file if has_file else None,
        url=url.strip() if has_url and url else None,
        category=typed_category,
    )
    return AnalyzeResponse(analysis_id=analysis_id)


def _is_stale(started_at: str | None) -> bool:
    """started_at 이후 STALE_THRESHOLD_SEC 초과면 죽은 분석으로 본다."""
    if not started_at:
        return False
    started = datetime.fromisoformat(started_at)
    return (datetime.now(UTC) - started).total_seconds() > STALE_THRESHOLD_SEC


@app.get(
    "/api/analyses/{analysis_id}/status", response_model=AnalysisStatusResponse
)
@limiter.limit("60/minute")
async def get_status(request: Request, analysis_id: str) -> AnalysisStatusResponse:
    """진행 상태 폴링 — 경량 단건 조회.

    status='analyzing'인데 Modal이 강제종료로 fail 기록조차 못한 stale row는 읽기
    시점에 failed로 정리한다(별도 cron 불필요).
    """
    s = await asyncio.to_thread(get_analysis_status, analysis_id)
    if s is None:
        raise HTTPException(status_code=404, detail="분석을 찾을 수 없습니다.")

    status = s["status"]
    error = s["error"]
    if status == "analyzing" and _is_stale(s["started_at"]):
        status = "failed"
        error = error or "분석이 시간 내에 완료되지 않았습니다."
        with suppress(Exception):
            await asyncio.to_thread(fail_analysis, analysis_id, s["video_id"], error)

    return AnalysisStatusResponse(status=status, progress=s["progress"], error=error)
