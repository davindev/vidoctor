"""Vidoctor Streamlit UI.

흐름:
1. 사이드바: 카테고리 선택 + 영상 업로드 + 분석 시작 / 이전 분석 리스트
2. 분석 실행: 영상 임시 저장 → graph 실행 → Supabase Storage 업로드 + DB 저장
3. 결과 표시: 5차원 카운트 + 차원별 이슈 리스트

영상 ≤ 50MB만 Supabase Storage 업로드 (Free tier). 초과 시 storage_path는 로컬 마커.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

import cv2
import streamlit as st
from pydantic import BaseModel

from vidoctor import __version__
from vidoctor.graph import Category, run_analysis
from vidoctor.repository import (
    complete_analysis,
    fail_analysis,
    get_analysis_findings,
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

# Supabase Free tier Storage 한도. 초과 영상은 메타만 저장하고 Storage 업로드는 skip.
_MAX_STORAGE_UPLOAD_BYTES = 50 * 1024 * 1024


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
            storage_path = f"local/{uploaded_file.name}"

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


def _render_findings(analysis_id: str) -> None:
    findings = get_analysis_findings(analysis_id)
    total = sum(len(events) for events in findings.values())

    st.metric("총 이슈", total)

    cols = st.columns(len(DIMENSION_LABEL))
    for col, (dim, label) in zip(cols, DIMENSION_LABEL.items(), strict=True):
        col.metric(label, len(findings.get(dim, [])))

    st.divider()

    for dim, label in DIMENSION_LABEL.items():
        events: list[BaseModel] = findings.get(dim, [])
        if not events:
            continue
        with st.expander(f"{label} ({len(events)}건)", expanded=False):
            rows = []
            for ev in events:
                d = ev.model_dump()
                rows.append({**d, "duration": d["end"] - d["start"]})
            st.dataframe(rows, use_container_width=True)


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
            _cached_list_analyses.clear()
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
    st.subheader("분석 결과")
    _render_findings(selected_id)
else:
    st.info("좌측에서 영상을 업로드하거나 이전 분석을 선택하세요.")
