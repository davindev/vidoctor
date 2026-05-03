"""Vidoctor Streamlit UI.

흐름:
1. 사이드바: 카테고리 선택 + 영상 업로드 + 분석 시작 / 이전 분석 리스트
2. 분석 실행: 영상 임시 저장 → graph 실행 → Supabase Storage 업로드 + DB 저장
3. 결과 표시: 5차원 카운트 + 차원별 이슈 리스트

영상 ≤ 50MB만 Supabase Storage 업로드 (Free tier). 초과 시 storage_path는 로컬 마커.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

import cv2
import streamlit as st
from pydantic import BaseModel

from vidoctor import __version__
from vidoctor.graph import Category, run_analysis
from vidoctor.graph.state import (
    ContentGapEvent,
    CPSEvent,
    DeadZoneEvent,
    FillerEvent,
    GazeEvent,
)
from vidoctor.repository import (
    LOCAL_STORAGE_PREFIX,
    complete_analysis,
    create_video_signed_url,
    delete_video_for_analysis,
    fail_analysis,
    get_analysis_findings,
    get_analysis_storage_path,
    insert_analysis,
    insert_video,
    list_analyses,
    upload_video_file,
)

CATEGORY_LABEL: dict[Category, str] = {
    "lecture": "강의",
    "vlog": "브이로그·인터뷰",
    "other": "기타",
}

DIMENSION_LABEL: dict[str, str] = {
    "filler": "Filler",
    "cps": "CPS",
    "dead_zone": "Dead Zone",
    "gaze": "Gaze",
    "content_gap": "Content Gap",
}

# 차원별 UI 버튼 라벨에 추가 표시할 Pydantic event 필드. 차원 추가 시 여기 한 줄.
_LABEL_EXTRA_FIELDS: dict[str, tuple[str, ...]] = {
    "filler": ("text",),
    "cps": ("kind",),
    "dead_zone": (),
    "gaze": ("direction",),
    "content_gap": ("description",),
}

# content_gap.description 같이 긴 텍스트는 버튼 라벨에서 truncate.
_LABEL_EXTRA_MAX_LEN = 30

# Supabase Free tier Storage 한도. 초과 영상은 메타만 저장하고 Storage 업로드는 skip.
_MAX_STORAGE_UPLOAD_BYTES = 50 * 1024 * 1024

# 5차원 finding 이벤트의 union — start/end/severity property 직접 접근을 타입 안전하게.
_FindingEvent = FillerEvent | CPSEvent | DeadZoneEvent | GazeEvent | ContentGapEvent


def _video_duration(path: Path) -> float | None:
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return frames / fps if fps > 0 else None
    finally:
        cap.release()


def _process_uploaded_video(uploaded_file, category: Category) -> str:  # noqa: ANN001
    """업로드 영상 → graph 실행 → Supabase 저장. analysis_id 반환."""
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = Path(tmp.name)

    try:
        duration = _video_duration(tmp_path)
        size = tmp_path.stat().st_size

        if size <= _MAX_STORAGE_UPLOAD_BYTES:
            with st.status("Storage 업로드 중...", expanded=False) as s:
                storage_path = upload_video_file(tmp_path, uploaded_file.name)
                s.update(state="complete")
        else:
            st.warning(
                f"영상 {size / 1024 / 1024:.0f}MB가 Free tier 50MB 한도 초과 — Storage 업로드 skip."
            )
            storage_path = f"{LOCAL_STORAGE_PREFIX}{uploaded_file.name}"

        video_id = insert_video(storage_path, category, duration)
        analysis_id = insert_analysis(video_id)

        try:
            with st.status("5차원 분석 실행 중 (1~3분)...", expanded=True) as s:
                state = asyncio.run(run_analysis(str(tmp_path), category))
                s.update(label="결과 저장 중...", state="running")
                complete_analysis(analysis_id, video_id, state)
                s.update(label="완료", state="complete")
        except Exception as e:
            fail_analysis(analysis_id, video_id, str(e))
            raise

        return analysis_id
    finally:
        tmp_path.unlink(missing_ok=True)


@st.cache_data(ttl=30)
def _cached_list_analyses(limit: int = 20) -> list[dict[str, Any]]:
    """사이드바 목록 — 새 분석 완료/선택 변경 시 명시적으로 invalidate."""
    return list_analyses(limit=limit)


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _event_button_label(dim: str, ev: _FindingEvent) -> str:
    extras: list[str] = []
    for field in _LABEL_EXTRA_FIELDS.get(dim, ()):
        v = getattr(ev, field, None)
        if v:
            text = str(v)
            extras.append(
                text if len(text) <= _LABEL_EXTRA_MAX_LEN
                else text[: _LABEL_EXTRA_MAX_LEN - 1] + "…"
            )
    suffix = f" · {' '.join(extras)}" if extras else ""
    return f"{_fmt_time(ev.start)}–{_fmt_time(ev.end)} · {ev.severity}{suffix}"


def _jump_key(analysis_id: str) -> str:
    return f"jump_to_{analysis_id}"


@st.cache_data(ttl=1800)
def _cached_video_url(analysis_id: str) -> str | None:
    """analysis_id → signed URL 또는 None. 매 rerun DB+API 호출 회피.

    signed URL TTL(3600s) 절반(1800s)으로 캐시해 만료 직전에도 안전 갱신.
    None 반환 케이스: Storage 미저장 또는 Storage에서 파일이 이미 삭제됨(stale state).
    """
    storage_path = get_analysis_storage_path(analysis_id)
    if storage_path is None:
        return None
    try:
        return create_video_signed_url(storage_path)
    except Exception:  # noqa: BLE001 - 404 / 네트워크 실패 모두 동일 처리
        return None


@st.cache_data(ttl=300)
def _cached_findings(analysis_id: str) -> dict[str, list[BaseModel]]:
    """findings는 분석 완료 후 immutable이라 짧은 TTL로 캐시."""
    return get_analysis_findings(analysis_id)


def _render_video_player(analysis_id: str) -> None:
    signed_url = _cached_video_url(analysis_id)
    if signed_url is None:
        st.info(
            "원본 영상을 재생할 수 없습니다 (Storage에 저장되지 않았거나 파일이 삭제됨)."
        )
        return
    start = int(st.session_state.get(_jump_key(analysis_id), 0))
    st.video(signed_url, start_time=start)


def _invalidate_caches() -> None:
    _cached_list_analyses.clear()
    _cached_video_url.clear()
    _cached_findings.clear()


def _confirm_key(analysis_id: str) -> str:
    return f"confirm_del_{analysis_id}"


def _render_delete_section(analysis_id: str) -> None:
    confirm_key = _confirm_key(analysis_id)
    if st.session_state.get(confirm_key):
        st.warning("영상과 모든 분석이 함께 삭제됩니다. 되돌릴 수 없습니다.")
        col_yes, col_no = st.columns(2)
        # 취소를 primary로 두어 안전한 기본 선택을 시각 강조 (destructive action 가드).
        if col_yes.button("정말 삭제", key=f"del_yes_{analysis_id}", type="secondary"):
            # 다른 탭에서 이미 삭제됐으면 LookupError, UI 동기화는 어차피 진행.
            with suppress(LookupError):
                delete_video_for_analysis(analysis_id)
            st.session_state[confirm_key] = False
            st.session_state["selected_analysis_id"] = None
            _invalidate_caches()
            st.rerun()
        if col_no.button("취소", key=f"del_no_{analysis_id}", type="primary"):
            st.session_state[confirm_key] = False
            st.rerun()
        return

    if st.button("🗑 영상 + 분석 삭제", key=f"del_{analysis_id}"):
        st.session_state[confirm_key] = True
        st.rerun()


def _render_findings_grid(analysis_id: str) -> None:
    findings = _cached_findings(analysis_id)
    total = sum(len(events) for events in findings.values())
    st.caption(f"총 {total}건")

    cols = st.columns(len(DIMENSION_LABEL))
    for col, (dim, label) in zip(cols, DIMENSION_LABEL.items(), strict=True):
        events: list[BaseModel] = findings.get(dim, [])
        with col:
            st.markdown(f"**{label}** · {len(events)}")
            for ev in events:
                fe = cast(_FindingEvent, ev)
                btn_label = _event_button_label(dim, fe)
                key = f"jmp-{analysis_id}-{dim}-{fe.start}"
                if st.button(btn_label, key=key, use_container_width=True):
                    st.session_state[_jump_key(analysis_id)] = fe.start
                    st.rerun()


def _format_analysis_label(a: dict[str, Any]) -> str:
    video = a.get("videos") or {}
    cat = video.get("category", "?")
    cat_label = CATEGORY_LABEL.get(cat, cat)
    started = (a.get("started_at") or "")[:16].replace("T", " ")
    suffix = " ⚠" if a.get("error") else ""
    return f"{cat_label} · {started}{suffix}"


# ---------------------------------------------------------------------------
# 페이지 본체
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Vidoctor", layout="wide")
st.title("Vidoctor")
st.caption(f"AI 영상 감수 에이전트 · v{__version__}")

if "selected_analysis_id" not in st.session_state:
    st.session_state["selected_analysis_id"] = None

with st.sidebar:
    st.subheader("새 분석")
    # st.selectbox 반환은 str. options이 Category 키 전체라 안전하게 cast.
    category = cast(
        Category,
        st.selectbox(
            "카테고리",
            options=list(CATEGORY_LABEL.keys()),
            format_func=lambda x: CATEGORY_LABEL[x],
        ),
    )
    uploaded = st.file_uploader("영상 파일", type=["mp4", "mov"])
    start = st.button("분석 시작", disabled=uploaded is None, type="primary")

    if start and uploaded is not None:
        try:
            analysis_id = _process_uploaded_video(uploaded, category)
            st.session_state["selected_analysis_id"] = analysis_id
            _invalidate_caches()
            st.rerun()
        except Exception as e:
            st.error(f"분석 실패: {e}")

    st.divider()
    st.subheader("이전 분석")
    for a in _cached_list_analyses(limit=20):
        if st.button(_format_analysis_label(a), key=f"sel-{a['id']}", use_container_width=True):
            st.session_state["selected_analysis_id"] = a["id"]
            st.rerun()


selected_id = st.session_state.get("selected_analysis_id")
if selected_id:
    _render_video_player(selected_id)
    st.divider()
    st.subheader("이슈 목록 — 클릭하면 영상이 해당 구간으로 이동")
    _render_findings_grid(selected_id)
    st.divider()
    _render_delete_section(selected_id)
else:
    st.info("좌측에서 영상을 업로드하거나 이전 분석을 선택하세요.")
