"""Vidoctor Streamlit UI.

흐름:
1. 사이드바: 카테고리 선택 + 영상 업로드 + 분석 시작 / 이전 분석 리스트
2. 분석 실행: 영상 임시 저장 → graph 실행 → R2 업로드 + Supabase DB 저장
3. 결과 표시: 5차원 카운트 + 차원별 이슈 리스트

업로드 한도는 `.streamlit/config.toml`의 `server.maxUploadSize`(300MB)에서 1차 가드.
"""

from __future__ import annotations

import asyncio
import shutil
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
    Suggestion,
    parse_finding_ref,
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

# GazeEvent.direction은 강사 입장(yaw<0=left=강사 좌측). 카메라가 강사를 비추므로 시청자
# 가시 영역에선 좌우 반전이 자연스러움. 한국어 시청자 입장 표기로 변환.
_GAZE_DIRECTION_LABEL: dict[str, str] = {
    "front": "정면",
    "left": "오른쪽",
    "right": "왼쪽",
    "up": "위",
    "down": "아래",
    "left_up": "오른쪽 위",
    "left_down": "오른쪽 아래",
    "right_up": "왼쪽 위",
    "right_down": "왼쪽 아래",
}

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
    """업로드 영상 → graph 실행 → R2 + Supabase DB 저장. analysis_id 반환."""
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    # getvalue()는 전체 bytes를 메모리에 올려 300MB 영상이면 피크 600MB+. chunked copy로 회피.
    uploaded_file.seek(0)
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(uploaded_file, tmp, length=8 * 1024 * 1024)
        tmp_path = Path(tmp.name)

    try:
        duration = _video_duration(tmp_path)

        with st.status("R2 업로드 중...", expanded=False) as s:
            storage_path = upload_video_file(tmp_path, uploaded_file.name)
            s.update(state="complete")

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
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _event_button_label(dim: str, ev: _FindingEvent) -> str:
    extras: list[str] = []
    for field in _LABEL_EXTRA_FIELDS.get(dim, ()):
        v = getattr(ev, field, None)
        if not v:
            continue
        if dim == "gaze" and field == "direction":
            text = _GAZE_DIRECTION_LABEL.get(str(v), str(v))
        else:
            text = str(v)
        extras.append(
            text if len(text) <= _LABEL_EXTRA_MAX_LEN
            else text[: _LABEL_EXTRA_MAX_LEN - 1] + "…"
        )
    suffix = f" · {' '.join(extras)}" if extras else ""
    return f"{_fmt_time(ev.start)}–{_fmt_time(ev.end)} · {ev.severity}{suffix}"


def _jump_key(analysis_id: str) -> str:
    return f"jump_to_{analysis_id}"


# 캐시 TTL은 signed URL expires_in 보다 충분히 짧아야 영상 재생 도중 URL이 만료되지
# 않는다 (st.video는 한 번 받은 URL로 브라우저가 이어서 range request). 1h 강의 재생
# 도중 만료를 피하려면 expires - TTL ≥ 1h 마진. 발급 비용은 사실상 0이라 길게 둔다.
_SIGNED_URL_EXPIRES_SEC = 7200
_VIDEO_URL_CACHE_TTL = 3000


@st.cache_data(ttl=_VIDEO_URL_CACHE_TTL)
def _cached_video_url(analysis_id: str) -> str | None:
    """analysis_id → signed URL 또는 None. 매 rerun DB+API 호출 회피.

    None 반환 = Storage 미저장(영구 상태). signed URL 발급 실패는 예외로 전파 —
    Streamlit `cache_data`가 예외에선 결과를 캐싱하지 않아 transient 실패가 캐시 TTL
    동안 None으로 굳어버리는 걸 방지. caller가 try/except로 분기 처리.
    """
    storage_path = get_analysis_storage_path(analysis_id)
    if storage_path is None:
        return None
    return create_video_signed_url(storage_path, expires_in=_SIGNED_URL_EXPIRES_SEC)


@st.cache_data(ttl=300)
def _cached_findings(analysis_id: str) -> dict[str, list[BaseModel]]:
    """findings는 분석 완료 후 immutable이라 짧은 TTL로 캐시."""
    return get_analysis_findings(analysis_id)


@st.cache_data(ttl=300)
def _cached_suggestions(analysis_id: str) -> list[Suggestion]:
    """suggestions도 분석 완료 후 immutable. priority 오름차순 정렬된 상태로 옴."""
    return get_analysis_suggestions(analysis_id)


@st.cache_data(ttl=300)
def _cached_analysis_meta(analysis_id: str) -> dict[str, Any]:
    return get_analysis_meta(analysis_id)


def _parse_iso(ts: str | None) -> float | None:
    """Supabase ISO timestamp → unix seconds. None·파싱 실패면 None."""
    if not ts:
        return None
    try:
        from datetime import datetime
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


def _render_cost_latency_card(analysis_id: str) -> None:
    meta = _cached_analysis_meta(analysis_id)
    cost = meta.get("cost_usd")
    started = _parse_iso(meta.get("started_at"))
    finished = _parse_iso(meta.get("finished_at"))
    latency_total = finished - started if started and finished else None

    cols = st.columns(2)
    cols[0].metric("LLM 비용", f"${cost:.4f}" if cost else "—")
    cols[1].metric("처리 시간", f"{latency_total:.1f}s" if latency_total is not None else "—")

    step_metrics = (meta.get("metadata") or {}).get("step_metrics") or []
    if step_metrics:
        with st.expander("LLM 호출 분리"):
            for sm in step_metrics:
                st.caption(
                    f"**{sm['step']}** · {sm['model']} · "
                    f"${sm['cost_usd']:.4f} · {sm['latency_sec']:.2f}s · "
                    f"prompt {sm['prompt_tokens']} / completion {sm['completion_tokens']} tok"
                )


def _render_video_player(analysis_id: str) -> None:
    try:
        signed_url = _cached_video_url(analysis_id)
    except Exception:  # noqa: BLE001 - signed URL 발급 실패는 transient, 다음 rerun 재시도
        st.info("원본 영상 파일을 찾을 수 없습니다 (R2에서 삭제됐거나 일시적 오류).")
        return
    if signed_url is None:
        st.info("원본 영상이 R2에 저장되지 않았습니다.")
        return
    start = int(st.session_state.get(_jump_key(analysis_id), 0))
    st.video(signed_url, start_time=start)


def _invalidate_caches() -> None:
    _cached_list_analyses.clear()
    _cached_video_url.clear()
    _cached_findings.clear()
    _cached_suggestions.clear()
    _cached_analysis_meta.clear()


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


def _resolve_finding_ref(
    findings: dict[str, list[BaseModel]], ref: str
) -> tuple[str, float] | None:
    """ref → (차원 한국어 라벨, 시작 시각). 무효 ref·인덱스 범위 밖이면 None."""
    parsed = parse_finding_ref(ref)
    if parsed is None:
        return None
    dim, idx = parsed
    events = findings.get(dim, [])
    if not 0 <= idx < len(events):
        return None
    ev = cast(_FindingEvent, events[idx])
    return DIMENSION_LABEL.get(dim, dim), ev.start


_REF_BUTTONS_PER_ROW = 6


def _render_suggestions(analysis_id: str) -> None:
    suggestions = _cached_suggestions(analysis_id)
    if not suggestions:
        st.caption("개선 제안 없음.")
        return
    findings = _cached_findings(analysis_id)
    for sug_idx, sug in enumerate(suggestions):
        with st.container(border=True):
            cols = st.columns([10, 1])
            cols[0].markdown(f"**{sug.text}**")
            cols[1].caption(f"P{sug.priority}")
            if sug.finding_refs:
                st.caption("근거 — 클릭하면 영상이 해당 구간으로 이동")
                for chunk_start in range(0, len(sug.finding_refs), _REF_BUTTONS_PER_ROW):
                    chunk = sug.finding_refs[chunk_start : chunk_start + _REF_BUTTONS_PER_ROW]
                    btn_cols = st.columns(_REF_BUTTONS_PER_ROW)
                    for i, ref in enumerate(chunk):
                        info = _resolve_finding_ref(findings, ref)
                        with btn_cols[i]:
                            if info is None:
                                st.caption(f"`{ref}` (?)")
                                continue
                            label, start = info
                            key = f"sug-jmp-{analysis_id}-{sug_idx}-{chunk_start + i}"
                            if st.button(
                                f"{label} {_fmt_time(start)}",
                                key=key,
                                use_container_width=True,
                            ):
                                st.session_state[_jump_key(analysis_id)] = start
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
    _render_cost_latency_card(selected_id)
    st.divider()
    st.subheader("개선 제안")
    _render_suggestions(selected_id)
    st.divider()
    st.subheader("이슈 목록 — 클릭하면 영상이 해당 구간으로 이동")
    _render_findings_grid(selected_id)
    st.divider()
    _render_delete_section(selected_id)
else:
    st.info("좌측에서 영상을 업로드하거나 이전 분석을 선택하세요.")
