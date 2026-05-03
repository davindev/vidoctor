"""Supabase repository: videos / analyses / findings CRUD.

graph 결과를 영구 저장해 UI에서 '이전 분석 다시 보기' 가능. service_role key로
단일 사용자 가정 (RLS 비활성, v1.1에서 인증 도입 시 정책 추가).

5차원 이벤트는 차원별 다른 필드(text/cps/direction/description...)를 갖지만 findings
테이블은 (start_sec/end_sec/severity/payload JSONB) 통합 스키마. 차원별 고유 필드는
payload JSONB로 직렬화·역직렬화한다.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

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
)

_log = logging.getLogger(__name__)

_DIM_TO_EVENT_CLASS: dict[Dimension, type[BaseModel]] = {
    "filler": FillerEvent,
    "cps": CPSEvent,
    "dead_zone": DeadZoneEvent,
    "gaze": GazeEvent,
    "content_gap": ContentGapEvent,
}

_STORAGE_BUCKET = "videos"

# Storage 미저장 영상의 storage_path 마커. 50MB 초과 등으로 Storage 업로드를 skip한 경우
# `f"{LOCAL_STORAGE_PREFIX}{filename}"` 형태로 DB에 기록해 추후 영상 재생 시 분기.
LOCAL_STORAGE_PREFIX = "local/"


@lru_cache(maxsize=1)
def _client() -> Client:
    """service_role key로 Supabase client 생성. 프로세스 수명 동안 1회 캐시."""
    settings = get_settings()
    return create_client(
        settings.supabase_url,
        settings.supabase_service_key.get_secret_value(),
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


_FINDING_TOP_FIELDS: frozenset[str] = frozenset({"start", "end", "severity"})


def _event_to_row(analysis_id: str, dim: str, event: BaseModel) -> dict[str, Any]:
    data = event.model_dump()
    payload = {k: v for k, v in data.items() if k not in _FINDING_TOP_FIELDS}
    return {
        "analysis_id": analysis_id,
        "dimension": dim,
        "start_sec": data["start"],
        "end_sec": data["end"],
        "severity": data.get("severity"),
        "payload": payload,
    }


def _row_to_event(row: dict[str, Any]) -> BaseModel:
    cls = _DIM_TO_EVENT_CLASS[row["dimension"]]
    kwargs: dict[str, Any] = {
        "start": row["start_sec"],
        "end": row["end_sec"],
        **(row.get("payload") or {}),
    }
    # severity는 DB nullable. None이면 이벤트 클래스의 기본값 사용.
    if row.get("severity") is not None:
        kwargs["severity"] = row["severity"]
    return cls(**kwargs)


def _collect_finding_rows(analysis_id: str, state: AnalysisState) -> list[dict[str, Any]]:
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
    """영상을 Storage에 업로드 후 storage_path(`bucket/name`) 반환. 동일 이름이면 덮어씀."""
    client = _client()
    with local_path.open("rb") as f:
        client.storage.from_(_STORAGE_BUCKET).upload(
            path=storage_name,
            file=f,
            file_options={"upsert": "true"},
        )
    return f"{_STORAGE_BUCKET}/{storage_name}"


def insert_video(
    storage_path: str,
    category: Category,
    duration_sec: float | None = None,
) -> str:
    res = (
        _client()
        .table("videos")
        .insert(
            {
                "storage_path": storage_path,
                "category": category,
                "duration_sec": duration_sec,
                "status": "analyzing",
            }
        )
        .execute()
    )
    return _first_row(res)["id"]


def update_video_status(video_id: str, status: str) -> None:
    _client().table("videos").update({"status": status}).eq("id", video_id).execute()


# ---------------------------------------------------------------------------
# analyses / findings
# ---------------------------------------------------------------------------


def insert_analysis(video_id: str) -> str:
    res = _client().table("analyses").insert({"video_id": video_id}).execute()
    return _first_row(res)["id"]


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
    """state의 5차원 이벤트를 findings 테이블에 bulk insert."""
    rows = _collect_finding_rows(analysis_id, state)
    if rows:
        _client().table("findings").insert(rows).execute()


# ---------------------------------------------------------------------------
# 종료 처리 — 호출자가 success/fail 두 경로만 신경 쓰면 되도록 묶음
# ---------------------------------------------------------------------------


def complete_analysis(
    analysis_id: str,
    video_id: str,
    state: AnalysisState,
    *,
    cost_usd: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """성공 종료: findings 저장 + analyses.finished_at + videos.status='completed'."""
    save_findings(analysis_id, state)
    finalize_analysis(analysis_id, cost_usd=cost_usd, metadata=metadata)
    update_video_status(video_id, "completed")


def fail_analysis(analysis_id: str, video_id: str, error: str) -> None:
    """실패 종료: analyses.error + videos.status='failed'."""
    finalize_analysis(analysis_id, error=error)
    update_video_status(video_id, "failed")


# ---------------------------------------------------------------------------
# 조회 (UI 사이드바·상세 페이지)
# ---------------------------------------------------------------------------


def list_analyses(limit: int = 20) -> list[dict[str, Any]]:
    """최근 분석 리스트. video 메타도 join으로 함께."""
    res = (
        _client()
        .table("analyses")
        .select(
            "id, started_at, finished_at, error, videos(category, storage_path, status)"
        )
        .order("started_at", desc=True)
        .limit(limit)
        .execute()
    )
    return _row_data(res)


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


def get_analysis_storage_path(analysis_id: str) -> str | None:
    """analysis_id → 연결된 video.storage_path. 영상 미저장(`local/...`)이면 None."""
    res = (
        _client()
        .table("analyses")
        .select("videos(storage_path)")
        .eq("id", analysis_id)
        .single()
        .execute()
    )
    data = cast(dict[str, Any] | None, res.data)
    if not data:
        return None
    video = cast(dict[str, Any] | None, data.get("videos"))
    if not video:
        return None
    storage_path = video.get("storage_path")
    if not isinstance(storage_path, str) or storage_path.startswith(LOCAL_STORAGE_PREFIX):
        return None
    return storage_path


def create_video_signed_url(storage_path: str, expires_in: int = 3600) -> str:
    """`bucket/path` 형식의 storage_path → 만료 시간 있는 signed URL.

    Storage bucket이 private이라 직접 URL은 401 — Supabase API로 서명된 URL 발급해
    Streamlit `st.video(url)`에 그대로 넘기면 브라우저가 streaming.
    """
    bucket, _, name = storage_path.partition("/")
    res = _client().storage.from_(bucket).create_signed_url(name, expires_in=expires_in)
    return cast(str, res["signedURL"])


def delete_video_for_analysis(analysis_id: str) -> None:
    """analysis가 속한 영상 + 같은 영상의 모든 analyses/findings/suggestions 일괄 삭제.

    videos row 삭제만으로 cascade(SQL FK on delete cascade)가 자동 처리. Storage
    파일이 있으면 그것도 함께 정리. Storage 삭제 실패는 silently 무시 — DB 정합성이
    더 중요하고, 고아 파일은 별도 정리 가능.
    """
    res = (
        _client()
        .table("analyses")
        .select("video_id, videos(storage_path)")
        .eq("id", analysis_id)
        .single()
        .execute()
    )
    data = cast(dict[str, Any] | None, res.data)
    if not data:
        raise LookupError(f"analysis not found: {analysis_id}")

    video_id = cast(str, data["video_id"])
    video = cast(dict[str, Any] | None, data.get("videos"))
    storage_path = video.get("storage_path") if video else None

    if isinstance(storage_path, str) and not storage_path.startswith(LOCAL_STORAGE_PREFIX):
        bucket, _, name = storage_path.partition("/")
        try:
            _client().storage.from_(bucket).remove([name])
        except Exception as e:  # noqa: BLE001
            # Storage 정리 실패해도 DB 삭제는 진행 — 객체 부재(404)·네트워크 오류 모두 동일 처리.
            # 고아 파일은 별도 정리 가능, DB 정합성이 우선.
            _log.warning(
                "storage cleanup failed: storage_path=%s err=%r", storage_path, e
            )

    _client().table("videos").delete().eq("id", video_id).execute()
