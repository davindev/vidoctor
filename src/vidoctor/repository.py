"""Supabase DB + Cloudflare R2 영상 storage — videos / analyses / findings CRUD.

영상 파일은 R2(S3 호환, egress 무료)에, 메타·findings·suggestions는 Supabase Postgres.
service_role key로 단일 사용자 가정 (RLS 비활성).

5차원 이벤트는 차원별 다른 필드(text/cps/direction/description...)를 가지지만 findings
테이블은 (start_sec/end_sec/payload JSONB) 통합 스키마 — 차원별 고유 필드는 payload
JSONB로 직렬화·역직렬화.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, TypeVar, cast

import boto3
import httpx
from botocore.config import Config
from postgrest.types import CountMethod
from pydantic import BaseModel
from supabase import Client, create_client

from vidoctor.config import get_settings
from vidoctor.graph.state import (
    DIM_TO_STATE_FIELD,
    AnalysisState,
    Category,
    ContentGapEvent,
    CPSEvent,
    DeadZoneEvent,
    Dimension,
    FillerEvent,
    GazeEvent,
    Suggestion,
)

_log = logging.getLogger(__name__)

_DIM_TO_EVENT_CLASS: dict[Dimension, type[BaseModel]] = {
    "filler": FillerEvent,
    "cps": CPSEvent,
    "dead_zone": DeadZoneEvent,
    "gaze": GazeEvent,
    "content_gap": ContentGapEvent,
}


@lru_cache(maxsize=1)
def _client() -> Client:
    """service_role key로 Supabase client 생성. 프로세스 수명 동안 1회 캐시."""
    settings = get_settings()
    return create_client(
        settings.supabase_url,
        settings.supabase_service_key.get_secret_value(),
    )


_T = TypeVar("_T")


def _retry_read(fn: Callable[..., _T]) -> Callable[..., _T]:
    """Supabase read를 stale 연결(RemoteProtocolError) 시 한 번 재시도하도록 감싼다.

    write는 재시도가 중복을 일으킬 수 있어 read 전용이다.
    """

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> _T:
        last: httpx.RemoteProtocolError | None = None
        for _ in range(2):
            try:
                return fn(*args, **kwargs)
            except httpx.RemoteProtocolError as e:
                last = e
        raise cast(httpx.RemoteProtocolError, last)

    return wrapper


@lru_cache(maxsize=1)
def _s3_client() -> Any:
    """R2(S3 호환) 클라이언트. region_name='auto'·SigV4는 R2 표준 컨벤션.

    boto3 client 자체는 thread-safe + 무거운 setup 없으나 lru_cache로 일관성 유지
    (Supabase client 캐시와 동일 패턴).
    """
    settings = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=settings.r2_endpoint,
        aws_access_key_id=settings.r2_access_key_id.get_secret_value(),
        aws_secret_access_key=settings.r2_secret_access_key.get_secret_value(),
        region_name="auto",
        # standard retry mode = 5xx + throttle + 네트워크 일시 오류에 대해 SDK 내부에서
        # 지수 backoff. 영구 실패(401/403/schema)는 첫 시도에 raise되어 retry 안 됨.
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )


def _row_data(res: Any) -> list[dict[str, Any]]:
    """Supabase APIResponse.data를 dict 리스트로 narrow.

    SDK 타입은 list[JSON union]이지만 우리 스키마상 row는 항상 dict — schema 불변을
    한 곳에서 cast로 표명. 호출자는 narrow된 타입 그대로 사용.
    """
    return cast(list[dict[str, Any]], res.data or [])


def _first_row(res: Any) -> dict[str, Any]:
    rows = _row_data(res)
    if not rows:
        raise RuntimeError("Supabase 응답에 데이터가 없습니다")
    return rows[0]


# ---------------------------------------------------------------------------
# 순수 변환 — graph event ↔ findings row
# ---------------------------------------------------------------------------


_FINDING_TOP_FIELDS: frozenset[str] = frozenset({"start", "end"})


def _event_to_row(analysis_id: str, dim: str, event: BaseModel) -> dict[str, Any]:
    data = event.model_dump()
    payload = {k: v for k, v in data.items() if k not in _FINDING_TOP_FIELDS}
    return {
        "analysis_id": analysis_id,
        "dimension": dim,
        "start_sec": data["start"],
        "end_sec": data["end"],
        "payload": payload,
    }


def _row_to_event(row: dict[str, Any]) -> BaseModel:
    cls = _DIM_TO_EVENT_CLASS[row["dimension"]]
    kwargs: dict[str, Any] = {
        "start": row["start_sec"],
        "end": row["end_sec"],
        **(row.get("payload") or {}),
    }
    return cls(**kwargs)


def _collect_finding_rows(analysis_id: str, state: AnalysisState) -> list[dict[str, Any]]:
    """state의 5차원 event를 findings 테이블 bulk insert row 리스트로 변환."""
    rows: list[dict[str, Any]] = []
    for dim, field in DIM_TO_STATE_FIELD.items():
        events = state.get(field, []) or []  # type: ignore[literal-required]
        for ev in events:
            rows.append(_event_to_row(analysis_id, dim, ev))
    return rows


# ---------------------------------------------------------------------------
# Storage / videos
# ---------------------------------------------------------------------------


def upload_video_file(local_path: Path, storage_name: str) -> str:
    """영상을 R2에 업로드 후 객체 키(storage_path) 반환. 동일 키면 덮어씀.

    boto3 `upload_fileobj`는 5MB 단위 자동 multipart라 300MB 영상도 메모리 폭발 없이
    스트리밍 업로드. transient 5xx/throttle은 boto3 Config(retries) 내장 backoff가 흡수.
    """
    settings = get_settings()
    with local_path.open("rb") as f:
        _s3_client().upload_fileobj(f, settings.r2_bucket, storage_name)
    return storage_name


def insert_video(
    storage_path: str,
    category: Category,
    duration_sec: float | None = None,
    filename: str | None = None,
) -> str:
    """videos row 생성 (status='analyzing') 후 id 반환.

    storage_path는 R2 객체 키(uuid 기반), filename은 표시용 원본 파일명.
    """
    res = (
        _client()
        .table("videos")
        .insert(
            {
                "storage_path": storage_path,
                "category": category,
                "duration_sec": duration_sec,
                "filename": filename,
                "status": "analyzing",
            }
        )
        .execute()
    )
    return _first_row(res)["id"]


def update_video_status(video_id: str, status: str) -> None:
    """videos.status를 갱신한다 (analyzing → completed/failed 전이)."""
    _client().table("videos").update({"status": status}).eq("id", video_id).execute()


# ---------------------------------------------------------------------------
# analyses / findings
# ---------------------------------------------------------------------------


def insert_analysis(video_id: str) -> str:
    """analyses row 생성 후 id 반환 (started_at은 DB default로 자동 채움)."""
    res = (
        _client()
        .table("analyses")
        .insert(
            {
                "video_id": video_id,
                "progress": {"phase": "running", "completed_nodes": []},
            }
        )
        .execute()
    )
    return _first_row(res)["id"]


def update_progress(analysis_id: str, node: str) -> None:
    """노드 완료 시 progress.completed_nodes에 멱등 append.

    LangGraph fan-out·retry로 같은 노드가 중복 들어올 수 있어 dedup (graph.run_analysis
    주석 참고). Supabase jsonb array의 atomic append가 제한적이라 read-merge-write로
    처리하며, Modal이 단일 분석을 직렬 처리하므로 동시 갱신 race는 없다.
    """
    res = (
        _client()
        .table("analyses")
        .select("progress")
        .eq("id", analysis_id)
        .maybe_single()
        .execute()
    )
    data = cast("dict[str, Any]", (res.data if res is not None else None) or {})
    progress = cast("dict[str, Any]", data.get("progress") or {})
    nodes = list(progress.get("completed_nodes") or [])
    if node not in nodes:
        nodes.append(node)
    progress["completed_nodes"] = nodes
    progress.setdefault("phase", "running")
    _client().table("analyses").update({"progress": progress}).eq(
        "id", analysis_id
    ).execute()


@_retry_read
def count_analyses_today() -> int:
    """오늘(UTC) 시작된 분석 수. 일일 글로벌 한도 체크용."""
    today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    res = (
        _client()
        .table("analyses")
        .select("id", count=CountMethod.exact)
        .gte("started_at", today_start.isoformat())
        .execute()
    )
    return res.count or 0


# Modal timeout(900s) + 콜드스타트 여유. 이 시간을 넘긴 'analyzing'은 죽은 것으로 본다.
STALE_THRESHOLD_SEC = 20 * 60


@_retry_read
def count_in_progress_analyses() -> int:
    """현재 진행 중(status='analyzing')이고 아직 stale 아닌 영상 수 — 동시성 가드용.

    Modal이 컨테이너 강제종료로 fail 기록을 못 한 stale row가 슬롯을 영구 점유하지
    않도록 created_at 하한을 같이 검사한다. 정확한 원자 카운트가 아니라 비용 폭증을
    막는 soft cap이다 (하드 상한은 Modal max_containers).
    """
    cutoff = datetime.now(UTC) - timedelta(seconds=STALE_THRESHOLD_SEC)
    res = (
        _client()
        .table("videos")
        .select("id", count=CountMethod.exact)
        .eq("status", "analyzing")
        .gte("created_at", cutoff.isoformat())
        .execute()
    )
    return res.count or 0


@_retry_read
def get_analysis_status(analysis_id: str) -> dict[str, Any] | None:
    """폴링용 경량 조회 — videos.status + analyses.progress/error/started_at + video_id.

    stale 판정·정리는 호출자(API)가 started_at으로 수행한다.
    """
    res = (
        _client()
        .table("analyses")
        .select("started_at, error, progress, video_id, videos(status, category)")
        .eq("id", analysis_id)
        .maybe_single()
        .execute()
    )
    data = cast("dict[str, Any] | None", res.data if res is not None else None)
    if not data:
        return None
    video = data.get("videos") or {}
    return {
        "status": video.get("status"),
        "category": video.get("category"),
        "progress": data.get("progress") or {},
        "error": data.get("error"),
        "started_at": data.get("started_at"),
        "video_id": data.get("video_id"),
    }


def finalize_analysis(
    analysis_id: str,
    *,
    error: str | None = None,
    cost_usd: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """분석 종료 시점에 finished_at + 부수 정보 기록."""
    payload: dict[str, Any] = {"finished_at": datetime.now(UTC).isoformat()}
    if error is not None:
        payload["error"] = error
    if cost_usd is not None:
        payload["cost_usd"] = cost_usd
    if metadata is not None:
        payload["metadata"] = metadata
    _client().table("analyses").update(payload).eq("id", analysis_id).execute()


def save_findings(analysis_id: str, state: AnalysisState) -> None:
    """state의 5차원 이벤트를 findings에 저장. 재시도 중복 방지로 기존 행을 비우고 insert(멱등)."""
    rows = _collect_finding_rows(analysis_id, state)
    _client().table("findings").delete().eq("analysis_id", analysis_id).execute()
    if rows:
        _client().table("findings").insert(rows).execute()


def save_suggestions(analysis_id: str, suggestions: list[Suggestion]) -> None:
    """suggestions를 저장. save_findings처럼 기존 행을 비우고 insert(멱등).
    finding_refs는 'filler:0' 같은 dim:idx 식별자."""
    _client().table("suggestions").delete().eq("analysis_id", analysis_id).execute()
    if not suggestions:
        return
    rows = [
        {
            "analysis_id": analysis_id,
            "text": s.text,
            "finding_refs": s.finding_refs,
        }
        for s in suggestions
    ]
    _client().table("suggestions").insert(rows).execute()


# ---------------------------------------------------------------------------
# 종료 처리 — 호출자가 success/fail 두 경로만 신경 쓰면 되도록 묶음
# ---------------------------------------------------------------------------


async def complete_analysis(
    analysis_id: str,
    video_id: str,
    state: AnalysisState,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """성공 종료: findings·suggestions 저장 + cost·step 메타 + analyses.finished_at +
    videos.status='completed'.

    4개 Supabase 호출은 모두 독립적이라 asyncio.gather로 동시 실행 (영상당 ~1초 절감).
    cost·latency는 state.step_metrics(LLM 노드들이 누적)에서 합산해 cost_usd로 저장하고,
    step별 분리 정보는 metadata.step_metrics JSON으로 보존 — UI가 분석 카드에 표시.
    """
    step_metrics = state.get("step_metrics", []) or []
    total_cost = sum(m.cost_usd for m in step_metrics)
    merged_metadata: dict[str, Any] = dict(metadata or {})
    failed_dimensions = state.get("failed_dimensions", []) or []
    if failed_dimensions:
        # add reducer는 concat만 해 중복이 쌓일 수 있어 dedup(등장 순서 유지).
        merged_metadata["failed_dimensions"] = list(dict.fromkeys(failed_dimensions))
    if step_metrics:
        merged_metadata["step_metrics"] = [
            {
                "step": m.step,
                "model": m.model,
                "cost_usd": round(m.cost_usd, 6),
                "latency_sec": round(m.latency_sec, 3),
                "prompt_tokens": m.prompt_tokens,
                "completion_tokens": m.completion_tokens,
            }
            for m in step_metrics
        ]

    # 저장 도중 stale 연결(RemoteProtocolError)로 실패하면 끝난 분석이 통째로 유실되므로
    # 1회 재시도한다. save_findings/save_suggestions가 멱등(삭제 후 insert)이라 중복 없음.
    for attempt in range(2):
        try:
            await asyncio.gather(
                asyncio.to_thread(save_findings, analysis_id, state),
                asyncio.to_thread(
                    save_suggestions, analysis_id, state.get("suggestions", []) or []
                ),
                asyncio.to_thread(
                    finalize_analysis,
                    analysis_id,
                    cost_usd=total_cost if step_metrics else None,
                    metadata=merged_metadata or None,
                ),
            )
            # completed 마킹은 위 저장이 끝난 뒤에. 폴링이 status='completed'를 보고 상세를
            # 가져가므로, 먼저 마킹되면 findings가 아직 없어 빈 결과를 받는다.
            await asyncio.to_thread(update_video_status, video_id, "completed")
            return
        except httpx.RemoteProtocolError:
            if attempt == 1:
                raise


def fail_analysis(analysis_id: str, video_id: str, error: str) -> None:
    """실패 종료: analyses.error + videos.status='failed'."""
    finalize_analysis(analysis_id, error=error)
    update_video_status(video_id, "failed")


# ---------------------------------------------------------------------------
# 조회 (UI 사이드바·상세 페이지)
# ---------------------------------------------------------------------------


@_retry_read
def list_analyses(limit: int = 20) -> list[dict[str, Any]]:
    """최근 분석 리스트. video 메타도 join으로 함께."""
    res = (
        _client()
        .table("analyses")
        .select(
            "id, started_at, finished_at, error, "
            "videos(category, storage_path, status, filename)"
        )
        .order("started_at", desc=True)
        .limit(limit)
        .execute()
    )
    return _row_data(res)


@_retry_read
def get_analysis_findings(analysis_id: str) -> dict[str, list[BaseModel]]:
    """findings를 차원별 list[Event]로 그룹화 — UI에서 직접 시각화할 수 있는 형태."""
    res = (
        _client()
        .table("findings")
        .select("*")
        .eq("analysis_id", analysis_id)
        .order("start_sec")
        .execute()
    )
    grouped: dict[str, list[BaseModel]] = {dim: [] for dim in DIM_TO_STATE_FIELD}
    for row in _row_data(res):
        dim = row["dimension"]
        if dim in grouped:
            grouped[dim].append(_row_to_event(row))
    return grouped


@_retry_read
def get_analysis_meta(analysis_id: str) -> dict[str, Any]:
    """UI 카드용 분석 메타 — cost_usd / 시작·종료 timestamp / step 분리 metadata.

    finished_at - started_at으로 wall-clock 처리 시간을 계산하고, metadata.step_metrics는
    LLM 호출 단계별 비용·token·latency를 분리 표시할 때 사용.
    """
    res = (
        _client()
        .table("analyses")
        .select("started_at, finished_at, cost_usd, metadata")
        .eq("id", analysis_id)
        .maybe_single()
        .execute()
    )
    return cast("dict[str, Any]", (res.data if res is not None else None) or {})


@_retry_read
def get_analysis_suggestions(analysis_id: str) -> list[Suggestion]:
    """suggestions를 저장 순서대로 반환 — LLM 출력 순서를 그대로 보존."""
    res = (
        _client()
        .table("suggestions")
        .select("text, finding_refs")
        .eq("analysis_id", analysis_id)
        .order("id")
        .execute()
    )
    return [
        Suggestion(
            text=row["text"],
            finding_refs=row.get("finding_refs") or [],
        )
        for row in _row_data(res)
    ]


@_retry_read
def get_analysis_video_meta(analysis_id: str) -> dict[str, Any] | None:
    """analysis_id → 연결된 video의 category/storage_path/duration_sec 메타.

    분석 결과 헤더(파일명 + 카테고리 pill)에 필요. 없으면 None.
    """
    res = (
        _client()
        .table("analyses")
        .select("videos(category, storage_path, duration_sec, filename)")
        .eq("id", analysis_id)
        .maybe_single()
        .execute()
    )
    data = cast("dict[str, Any] | None", res.data if res is not None else None)
    if not data:
        return None
    return cast(dict[str, Any] | None, data.get("videos"))


@_retry_read
def get_analysis_storage_path(analysis_id: str) -> str | None:
    """analysis_id → 연결된 video.storage_path(R2 객체 키). 없으면 None."""
    res = (
        _client()
        .table("analyses")
        .select("videos(storage_path)")
        .eq("id", analysis_id)
        .maybe_single()
        .execute()
    )
    data = cast("dict[str, Any] | None", res.data if res is not None else None)
    if not data:
        return None
    video = cast(dict[str, Any] | None, data.get("videos"))
    if not video:
        return None
    storage_path = video.get("storage_path")
    return storage_path if isinstance(storage_path, str) and storage_path else None


def create_video_signed_url(storage_path: str, expires_in: int = 3600) -> str:
    """R2 객체 키 → 만료 시간 있는 signed URL.

    버킷이 private이라 직접 URL은 401 — boto3 `generate_presigned_url`로 서명된 URL을
    만들어 클라이언트가 range request로 streaming 가능.
    """
    settings = get_settings()
    return cast(
        str,
        _s3_client().generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.r2_bucket, "Key": storage_path},
            ExpiresIn=expires_in,
        ),
    )


def delete_video_for_analysis(analysis_id: str) -> None:
    """analysis가 속한 영상 + 같은 영상의 모든 analyses/findings/suggestions 일괄 삭제.

    videos row 삭제만으로 cascade(SQL FK on delete cascade)가 자동 처리. R2 객체가
    있으면 함께 정리. R2 삭제 실패는 silently 무시 — DB 정합성이 더 중요하고, 고아
    객체는 별도 정리 가능.
    """
    res = (
        _client()
        .table("analyses")
        .select("video_id, videos(storage_path)")
        .eq("id", analysis_id)
        .maybe_single()
        .execute()
    )
    data = cast("dict[str, Any] | None", res.data if res is not None else None)
    if not data:
        raise LookupError(f"분석을 찾을 수 없습니다: {analysis_id}")

    video_id = cast(str, data["video_id"])
    video = cast(dict[str, Any] | None, data.get("videos"))
    storage_path = video.get("storage_path") if video else None

    if isinstance(storage_path, str) and storage_path:
        try:
            settings = get_settings()
            _s3_client().delete_object(Bucket=settings.r2_bucket, Key=storage_path)
        except Exception as e:  # noqa: BLE001
            # R2 정리 실패해도 DB 삭제는 진행 — 객체 부재(404)·네트워크 오류 모두 동일 처리.
            _log.warning(
                "R2 정리 실패: storage_path=%s err=%r", storage_path, e
            )

    _client().table("videos").delete().eq("id", video_id).execute()
