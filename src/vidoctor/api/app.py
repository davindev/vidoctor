"""FastAPI app — Next.js 프론트엔드와 통신하는 HTTP 엔드포인트.

엔드포인트:
- GET  /api/analyses                    — 이전 분석 리스트
- GET  /api/analyses/{id}              — 단건 (meta + findings + suggestions)
- GET  /api/analyses/{id}/video-url    — R2 signed URL
- DELETE /api/analyses/{id}            — 영상 + 모든 분석·findings·suggestions 삭제
- POST /api/analyze                    — multipart 업로드 → SSE 스트림으로 진행 이벤트
                                          (`started`/`uploaded`/`node`/`complete`/`error`)

CORS는 dev 단계에서 Next.js dev server(localhost:3000)를 허용.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from vidoctor.api.youtube import YouTubeIngestError, download_youtube
from vidoctor.graph import Category, run_analysis
from vidoctor.graph.state import (
    CATEGORY_DIMENSIONS,
    DIM_TO_STATE_FIELD,
    AnalysisState,
)
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

_log = logging.getLogger(__name__)

app = FastAPI(title="Vidoctor API", version="0.1.0")

# Dev 단계 CORS — Next.js dev (localhost:3000) + 동일 호스트 prod 빌드 모두 허용.
# prod 배포 시 origin allowlist 좁히는 것 권장.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas — 프론트엔드 클라이언트가 직접 import하는 타입 단일 진실
# ---------------------------------------------------------------------------


class AnalysisListItem(BaseModel):
    id: str
    started_at: str | None
    finished_at: str | None
    error: str | None
    category: str | None
    storage_path: str | None
    status: str | None


class FindingItem(BaseModel):
    dimension: str
    start: float
    end: float
    payload: dict[str, Any]


class SuggestionItem(BaseModel):
    text: str
    finding_refs: list[str]


class StepMetric(BaseModel):
    step: str
    model: str
    cost_usd: float
    latency_sec: float
    prompt_tokens: int
    completion_tokens: int


class SpeakerTurn(BaseModel):
    start: float
    end: float
    speaker: str
    word_count: int
    text_preview: str


class SpeakerDiarization(BaseModel):
    main_speaker: str
    durations: dict[str, float]
    turns: list[SpeakerTurn]


class AnalysisDetail(BaseModel):
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
    url: str | None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/analyses", response_model=list[AnalysisListItem])
async def list_recent_analyses(limit: int = 20) -> list[AnalysisListItem]:
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
    meta = await asyncio.to_thread(get_analysis_meta, analysis_id)
    if not meta:
        raise HTTPException(status_code=404, detail="analysis not found")

    findings_raw = await asyncio.to_thread(get_analysis_findings, analysis_id)
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

    suggestions_raw = await asyncio.to_thread(get_analysis_suggestions, analysis_id)
    suggestions = [
        SuggestionItem(text=s.text, finding_refs=s.finding_refs) for s in suggestions_raw
    ]

    metadata = cast(dict[str, Any], meta.get("metadata") or {})
    step_metrics = [StepMetric(**sm) for sm in metadata.get("step_metrics", [])]
    diar_raw = metadata.get("speaker_diarization")
    diar = SpeakerDiarization(**diar_raw) if diar_raw else None

    video_meta = await asyncio.to_thread(get_analysis_video_meta, analysis_id)

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
    storage_path = await asyncio.to_thread(get_analysis_storage_path, analysis_id)
    if storage_path is None:
        return VideoUrlResponse(url=None)
    url = await asyncio.to_thread(create_video_signed_url, storage_path, 7200)
    return VideoUrlResponse(url=url)


@app.delete("/api/analyses/{analysis_id}")
async def delete_analysis(analysis_id: str) -> dict[str, str]:
    try:
        await asyncio.to_thread(delete_video_for_analysis, analysis_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# POST /api/analyze — multipart 업로드 + SSE 진행 스트림
# ---------------------------------------------------------------------------


def _sse(event: str, data: dict[str, Any]) -> str:
    """Server-Sent Events 단일 이벤트 라인 — `event:` + `data:` + 빈 줄."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _save_upload_to_tmp(uploaded: UploadFile) -> tuple[Path, str]:
    """UploadFile을 청크 단위로 임시 파일에 떨궈 메모리 폭발 회피."""
    suffix = Path(uploaded.filename or "video.mp4").suffix or ".mp4"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        while chunk := await uploaded.read(8 * 1024 * 1024):
            tmp.write(chunk)
        return Path(tmp.name), (uploaded.filename or "video.mp4")


async def _analyze_stream(
    *,
    upload: UploadFile | None,
    url: str | None,
    category: Category,
) -> AsyncIterator[str]:
    """영상(파일 업로드 또는 유튜브 URL)을 임시 파일에 떨군 뒤 R2 업로드 → graph 실행 → DB 저장.

    각 단계 시점에 SSE 이벤트를 yield. 클라이언트는 `EventSource` 또는 fetch+ReadableStream
    으로 구독해 진행 그래프를 갱신.
    """
    tmp_path: Path | None = None
    analysis_id: str | None = None
    video_id: str | None = None

    try:
        # 1) 소스 → 로컬 tmp 파일. URL 경로는 다운로드 단계 표시를 위한 별도 SSE phase.
        if url is not None:
            yield _sse("status", {"phase": "downloading"})
            try:
                tmp_path, title = await download_youtube(url)
            except YouTubeIngestError as e:
                yield _sse("error", {"message": str(e), "analysis_id": None})
                return
            filename = f"{title}.mp4"
            # 다운로드가 끝나야 비로소 영상 제목을 알 수 있다 — 헤더 placeholder
            # "유튜브 URL"을 실제 제목으로 교체하도록 클라이언트에 통지.
            yield _sse("metadata", {"filename": filename})
        else:
            if upload is None:
                # endpoint XOR 검증이 빠졌을 때만 발생하는 invariant 위반.
                raise RuntimeError("upload XOR url invariant broken")
            tmp_path, filename = await _save_upload_to_tmp(upload)

        yield _sse("status", {"phase": "uploading"})
        storage_path = await asyncio.to_thread(upload_video_file, tmp_path, filename)

        # videos·analyses row 만들고 client에 analysis_id 통지 — 이후 graph가 끝나기 전에
        # client가 새로고침해도 in-progress row를 폴링할 수 있게.
        video_id = await asyncio.to_thread(insert_video, storage_path, category, None)
        analysis_id = await asyncio.to_thread(insert_analysis, video_id)
        yield _sse("started", {"analysis_id": analysis_id})
        yield _sse("uploaded", {})

        # graph 노드 완료 이벤트를 SSE로 relay. `on_node_complete`는 sync callback이라
        # asyncio.Queue로 producer/consumer 분리. graph 종료 시 sentinel(None)을 큐에
        # 넣어 main loop이 깔끔하게 빠져나오게 함 — graph_task와 queue.get()을 동시
        # await하던 이전 패턴은 race·cancel-leak 위험.
        _SENTINEL = object()
        loop = asyncio.get_running_loop()
        node_queue: asyncio.Queue[Any] = asyncio.Queue()

        def _on_node(name: str) -> None:
            loop.call_soon_threadsafe(
                node_queue.put_nowait, ("node", {"name": name})
            )

        async def _drive_graph() -> AnalysisState:
            try:
                return await run_analysis(
                    str(tmp_path), category, on_node_complete=_on_node
                )
            finally:
                node_queue.put_nowait(_SENTINEL)

        graph_task = asyncio.create_task(_drive_graph())

        while True:
            item = await node_queue.get()
            if item is _SENTINEL:
                break
            event_name, payload = item
            yield _sse(event_name, payload or {})

        graph_state = await graph_task

        await asyncio.to_thread(complete_analysis, analysis_id, video_id, graph_state)
        yield _sse("complete", {"analysis_id": analysis_id})

    except (asyncio.CancelledError, GeneratorExit):
        # 클라이언트 disconnect — DB row가 in-progress로 영구히 남지 않게 fail 처리.
        if analysis_id and video_id:
            with suppress(Exception):
                await asyncio.to_thread(
                    fail_analysis, analysis_id, video_id, "client disconnected"
                )
        raise
    except Exception as e:  # noqa: BLE001
        _log.exception("analysis pipeline failed")
        if analysis_id and video_id:
            with suppress(Exception):
                await asyncio.to_thread(fail_analysis, analysis_id, video_id, str(e))
        yield _sse("error", {"message": str(e), "analysis_id": analysis_id})
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


@app.post("/api/analyze")
async def analyze(
    category: str = Form(...),
    file: UploadFile | None = File(default=None),
    url: str | None = Form(default=None),
) -> StreamingResponse:
    if category not in CATEGORY_DIMENSIONS:
        raise HTTPException(status_code=400, detail=f"unknown category: {category}")
    # 파일 XOR URL — 둘 다 들어오거나 둘 다 비면 400.
    has_file = file is not None and (file.filename or "") != ""
    has_url = url is not None and url.strip() != ""
    if has_file == has_url:  # 둘 다 True 또는 둘 다 False
        raise HTTPException(
            status_code=400, detail="file 또는 url 중 정확히 하나를 제공해야 합니다."
        )
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
