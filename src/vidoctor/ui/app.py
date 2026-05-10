"""Vidoctor Streamlit UI.

흐름 (3-state):
1. **idle** — 메인에 카테고리·영상 업로드·분석 시작 폼. 사이드바엔 이전 분석 목록.
2. **analyzing** — 메인에 5차원 그래프(transcribe → fan-out → suggestions). 사이드바·폼 모두
   비활성. on_node_complete 콜백마다 그래프 재렌더.
3. **result** — 메인에 영상 플레이어 + 비용·latency 카드 + 제안 + finding 그리드.

업로드 한도는 `.streamlit/config.toml`의 `server.maxUploadSize`(300MB)에서 1차 가드.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

import cv2
import streamlit as st
from pydantic import BaseModel

from vidoctor import __version__
from vidoctor.config import ROOT
from vidoctor.graph import Category, run_analysis
from vidoctor.graph.pipeline import detector_node_name
from vidoctor.graph.state import (
    CATEGORY_DIMENSIONS,
    ContentGapEvent,
    CPSEvent,
    DeadZoneEvent,
    Dimension,
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
    "filler": "추임새",
    "cps": "말 속도",
    "dead_zone": "정적 구간",
    "gaze": "시선 이탈",
    "content_gap": "내용 불일치",
}

# cps `too_fast`/`too_slow`는 detector raw 출력. UI에선 한국어로 변환.
_CPS_KIND_LABEL: dict[str, str] = {"too_fast": "빠름", "too_slow": "느림"}

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

# 5차원 finding 이벤트의 union — start/end property 직접 접근을 타입 안전하게.
_FindingEvent = FillerEvent | CPSEvent | DeadZoneEvent | GazeEvent | ContentGapEvent

# detector 외 진행률 라벨 — detector 라벨은 DIMENSION_LABEL과 결합해 derive하므로
# 차원 추가 시엔 DIMENSION_LABEL만 손대면 된다.
_NON_DETECTOR_NODE_LABEL: dict[str, str] = {
    "transcribe": "음성 전사",
    "generate_suggestions": "개선 제안 생성",
}


def _node_label(node: str) -> str:
    for dim, label in DIMENSION_LABEL.items():
        if node == detector_node_name(cast(Dimension, dim)):
            return f"{label} 검출"
    return _NON_DETECTOR_NODE_LABEL.get(node, node)


def _expected_node_order(category: Category) -> list[str]:
    detectors = [detector_node_name(d) for d in CATEGORY_DIMENSIONS[category]]
    return ["transcribe", *detectors, "generate_suggestions"]


# 분석 진행 그래프 — 영상 업로드 → 음성 전사 → 활성 detectors fan-out → 개선 제안 fan-in.
# 노드 상태 3종: done(완료, 노란 채움), running(진행 중, 검정 테두리 + 펄스 애니메이션),
# pending(대기, 회색). on_upload_complete · on_node_complete 콜백마다 st.empty()의
# markdown(unsafe_allow_html=True)로 재렌더 — 매 렌더마다 done 노드는 fade-in 애니메이션,
# running 노드는 무한 pulse 애니메이션.
_GRAPH_CSS = """
<style>
.vid-node-running rect { animation: vidNodePulse 1.6s ease-in-out infinite; }
.vid-node-done rect, .vid-node-done text { animation: vidNodeAppear 0.4s ease-out; }
.vid-edge-done { animation: vidEdgeAppear 0.3s ease-out; }
@keyframes vidNodePulse {
  0%, 100% { opacity: 0.7; }
  50% { opacity: 1; }
}
@keyframes vidNodeAppear {
  from { opacity: 0; transform: scale(0.96); }
  to { opacity: 1; transform: scale(1); }
}
.vid-node-done g, .vid-node-done rect, .vid-node-done text { transform-box: fill-box; transform-origin: center; }
@keyframes vidEdgeAppear {
  from { stroke-dashoffset: 60; opacity: 0; }
  to { stroke-dashoffset: 0; opacity: 1; }
}
</style>
"""


def _render_dimension_graph(
    completed: set[str], category: Category, upload_done: bool
) -> str:
    detectors = [detector_node_name(d) for d in CATEGORY_DIMENSIONS[category]]
    n = len(detectors)

    # 4-column layout: upload | transcribe | detectors (vertical fan) | suggestions
    width, height = 760, 280
    x_up, x_tr, x_det, x_sug = 70, 230, 440, 620
    y_top, y_bot = 40, 240
    y_step = (y_bot - y_top) / (n - 1) if n > 1 else 0
    y_detectors = (
        [y_top + y_step * i for i in range(n)] if n > 1 else [(y_top + y_bot) / 2]
    )
    y_center = (y_top + y_bot) / 2

    transcribe_done = "transcribe" in completed
    suggestions_done = "generate_suggestions" in completed

    def node_state(is_done: bool, upstream_ok: bool) -> str:
        if is_done:
            return "done"
        return "running" if upstream_ok else "pending"

    parts: list[str] = []

    # 엣지 먼저 (노드가 위에 겹치도록)
    parts.append(_graph_edge(x_up + 50, y_center, x_tr - 50, y_center, upload_done))
    for i, name in enumerate(detectors):
        parts.append(
            _graph_edge(
                x_tr + 50, y_center, x_det - 50, y_detectors[i], transcribe_done
            )
        )
    for i, name in enumerate(detectors):
        parts.append(
            _graph_edge(
                x_det + 50, y_detectors[i], x_sug - 50, y_center, name in completed
            )
        )

    # 노드들
    parts.append(_graph_node(x_up, y_center, node_state(upload_done, True), "영상 업로드"))
    parts.append(
        _graph_node(
            x_tr, y_center, node_state(transcribe_done, upload_done), "음성 전사"
        )
    )
    for i, name in enumerate(detectors):
        parts.append(
            _graph_node(
                x_det,
                y_detectors[i],
                node_state(name in completed, transcribe_done),
                _node_label(name),
            )
        )
    all_detectors_done = all(d in completed for d in detectors) and detectors
    parts.append(
        _graph_node(
            x_sug,
            y_center,
            node_state(suggestions_done, bool(all_detectors_done)),
            "개선 제안",
        )
    )

    return (
        _GRAPH_CSS
        + f'<div style="display:flex;justify-content:center;padding:24px 0;">'
        f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet" '
        f'style="width:100%;max-width:{width}px;height:auto;">'
        f'{"".join(parts)}'
        f"</svg></div>"
    )


def _graph_node(x: float, y: float, state: str, label: str) -> str:
    if state == "done":
        fill, border, text_color, border_width = "#FAEFE7", "#D97757", "#1F1A17", 1.6
        klass = "vid-node-done"
    elif state == "running":
        fill, border, text_color, border_width = "#FFFFFF", "#D97757", "#1F1A17", 1.6
        klass = "vid-node-running"
    else:
        fill, border, text_color, border_width = "#F5EFE6", "#EAE2D6", "#9A8E86", 1.2
        klass = "vid-node-pending"
    return (
        f'<g class="{klass}">'
        f'<rect x="{x - 60}" y="{y - 22}" width="120" height="44" rx="22" '
        f'fill="{fill}" stroke="{border}" stroke-width="{border_width}"/>'
        f'<text x="{x}" y="{y + 5}" text-anchor="middle" font-size="13" '
        f"font-family=\"'Pretendard',-apple-system,sans-serif\" "
        f'font-weight="500" fill="{text_color}">{label}</text>'
        f"</g>"
    )


def _graph_edge(x1: float, y1: float, x2: float, y2: float, done: bool) -> str:
    stroke = "#D97757" if done else "#DBD0BF"
    width = 2.0 if done else 1.4
    dash = 'stroke-dasharray="60 60"' if done else 'stroke-dasharray="4 4"'
    klass = "vid-edge-done" if done else ""
    cx = (x1 + x2) / 2
    return (
        f'<path class="{klass}" d="M {x1} {y1} C {cx} {y1}, {cx} {y2}, {x2} {y2}" '
        f'stroke="{stroke}" stroke-width="{width}" fill="none" {dash}/>'
    )


def _video_duration(path: Path) -> float | None:
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return frames / fps if fps > 0 else None
    finally:
        cap.release()


def _run_analysis_pipeline(
    uploaded_file,  # noqa: ANN001
    category: Category,
    on_upload_complete: Callable[[], None],
    on_node: Callable[[str], None],
) -> str:
    """업로드 영상 → R2 업로드 + graph 실행 + DB 저장. analysis_id 반환.

    호출자(UI)가 진행 그래프 placeholder를 관리하고, 업로드·노드 완료 콜백을 통해 매번
    SVG를 재렌더한다. 이 함수는 R2 / DB / graph orchestration만 책임.
    """
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    # getvalue()는 전체 bytes를 메모리에 올려 300MB 영상이면 피크 600MB+. chunked copy로 회피.
    uploaded_file.seek(0)
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(uploaded_file, tmp, length=8 * 1024 * 1024)
        tmp_path = Path(tmp.name)

    try:
        duration = _video_duration(tmp_path)
        storage_path = upload_video_file(tmp_path, uploaded_file.name)
        on_upload_complete()
        video_id = insert_video(storage_path, category, duration)
        analysis_id = insert_analysis(video_id)

        try:
            state = asyncio.run(
                run_analysis(str(tmp_path), category, on_node_complete=on_node)
            )
            complete_analysis(analysis_id, video_id, state)
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
        elif dim == "cps" and field == "kind":
            text = _CPS_KIND_LABEL.get(str(v), str(v))
        else:
            text = str(v)
        extras.append(
            text if len(text) <= _LABEL_EXTRA_MAX_LEN
            else text[: _LABEL_EXTRA_MAX_LEN - 1] + "…"
        )
    suffix = f" · {' '.join(extras)}" if extras else ""
    return f"{_fmt_time(ev.start)}–{_fmt_time(ev.end)}{suffix}"


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
    """suggestions도 분석 완료 후 immutable. LLM이 출력한 순서를 유지해 표시."""
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
            st.markdown(f"**{sug.text}**")
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


_HISTORY_LABEL_MAX = 36


def _format_analysis_label(a: dict[str, Any]) -> str:
    """이전 분석 버튼 라벨 — 파일명(카테고리) 형식. 긴 파일명은 truncate."""
    video = a.get("videos") or {}
    cat = video.get("category", "?")
    cat_label = CATEGORY_LABEL.get(cat, cat)
    storage_path = str(video.get("storage_path") or "?")
    name = storage_path.rsplit("/", 1)[-1]  # bucket prefix 있는 경우 대비
    if len(name) > _HISTORY_LABEL_MAX:
        name = name[: _HISTORY_LABEL_MAX - 1] + "…"
    suffix = " ⚠" if a.get("error") else ""
    return f"{name} ({cat_label}){suffix}"


# ---------------------------------------------------------------------------
# 페이지 본체
# ---------------------------------------------------------------------------

_LOGO_PATH = str(ROOT / "assets" / "logo" / "vidoctor.svg")

st.set_page_config(page_title="Vidoctor", page_icon=_LOGO_PATH, layout="wide")


def _inject_global_styles() -> None:
    """Claude warm editorial 디자인 시스템 — 테라코타 액센트 + 웜 잉크 + 크림 배경.

    디자인 토큰은 design.md(Claude) 무드를 그대로 가져옴. Pretendard(UI) + Source Serif 4
    (헤딩·세리프 액센트) + JetBrains Mono(필드 번호·메타) 트리오. Streamlit chrome은
    `#MainMenu`·deploy 등 내부 항목만 surgical 숨김 (사이드바 토글 보존), Material 아이콘
    폰트는 cascade 우선 보호.
    """
    st.markdown(
        """
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/[email protected]/dist/web/static/pretendard.min.css');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,500;0,8..60,600;1,8..60,400&display=swap');

:root {
    --accent: #D97757;
    --accent-strong: #C15F3C;
    --accent-soft: #F2DDD0;
    --accent-tint: #FAEFE7;
    --ink: #1F1A17;
    --ink-2: #2E2722;
    --ink-3: #6B5F58;
    --ink-4: #9A8E86;
    --line: #EAE2D6;
    --line-2: #DBD0BF;
    --bg: #FAF7F2;
    --surface: #FFFFFF;
    --surface-tint: #F5EFE6;
    --danger: #B5483D;
    --mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
    --sans: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Apple SD Gothic Neo', system-ui, sans-serif;
    --serif: 'Source Serif 4', 'Source Serif Pro', Georgia, serif;
}

html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {
    font-family: var(--sans);
    background: var(--bg);
    color: var(--ink);
    -webkit-font-smoothing: antialiased;
    font-feature-settings: "ss01", "cv11";
}
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"],
.stButton button, .stTextInput input, .stTextArea textarea, [data-baseweb="select"] {
    font-family: var(--sans);
}
[data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded, .material-symbols-outlined {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
}

/* Streamlit chrome — 사이드바 토글 보존, 그 외 숨김 */
#MainMenu, .stDeployButton, [data-testid="stStatusWidget"], footer, [data-testid="stDecoration"] {
    display: none !important;
}
[data-testid="stHeader"] { background: transparent; }

/* Sidebar — Claude 디자인의 brand + history 섹션 */
[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 0; }

.vid-brand {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 0 18px;
    border-bottom: 1px solid var(--line);
    margin-bottom: 14px;
}
.vid-brand-mark {
    width: 30px; height: 30px;
    background: var(--accent); color: #fff;
    display: grid; place-items: center;
    font-family: var(--serif); font-weight: 500; font-size: 16px;
    border-radius: 50%;
}
.vid-brand-name {
    font-family: var(--serif); font-weight: 500; font-size: 20px;
    letter-spacing: -0.02em; color: var(--ink);
}
.vid-sidebar-title {
    font-size: 11px; font-weight: 500;
    letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--ink-4);
    padding: 4px 0 8px;
}

/* Sidebar history buttons — 거의 흰색 hover, 따뜻한 톤 */
[data-testid="stSidebar"] .stButton > button {
    width: 100%; text-align: left;
    background: transparent;
    border: 1px solid transparent;
    padding: 10px 12px;
    border-radius: 6px;
    color: var(--ink);
    font-weight: 500;
    font-size: 13px;
    transition: background 0.12s ease, border-color 0.12s ease;
}
[data-testid="stSidebar"] .stButton > button:hover:not(:disabled) {
    background: #FBF8F2;
    border-color: transparent;
    transform: none;
    box-shadow: none;
}
[data-testid="stSidebar"] .stButton > button:disabled { opacity: 0.5; }

/* Main 진입 fade-in + max-width 제약 */
[data-testid="stAppViewBlockContainer"] {
    max-width: 880px !important;
    padding-top: 56px !important;
    padding-bottom: 80px !important;
    animation: vidPageFade 0.35s ease-out;
}
@keyframes vidPageFade {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
}

/* 헤딩 */
.vid-h1 {
    font-family: var(--serif);
    font-size: 46px;
    font-weight: 600;
    letter-spacing: -0.028em;
    line-height: 1.05;
    margin: 0 0 16px;
    color: var(--ink);
}
.vid-h1 .vid-accent { color: var(--accent); font-weight: 600; }
.vid-lede {
    font-size: 16px; color: var(--ink-3); line-height: 1.65;
    max-width: 56ch; margin: 0 0 44px;
}
.vid-lede b { color: var(--ink); font-weight: 600; }

/* 필드 번호 라벨 */
.vid-field-head {
    display: flex; align-items: center; gap: 10px;
    margin: 8px 0 14px;
}
.vid-field-num {
    font-family: var(--mono);
    font-size: 11px; font-weight: 500;
    color: var(--accent);
    letter-spacing: 0.04em;
}
.vid-field-label {
    font-size: 14.5px; font-weight: 600; color: var(--ink);
}
.vid-field-hint {
    font-size: 12px; color: var(--ink-4);
    margin-left: auto;
    font-style: italic;
    font-family: var(--serif);
}

.vid-submit-hint {
    font-size: 12px; color: var(--ink-4); margin-left: 12px;
}
.vid-submit-hint .ok { color: var(--accent); }

/* selectbox */
[data-baseweb="select"] > div {
    border-radius: 6px;
    border: 1px solid var(--line-2);
    background: var(--surface);
    transition: border-color 0.12s ease, box-shadow 0.12s ease;
}
[data-baseweb="select"] > div:hover { border-color: var(--ink-3); }
[data-baseweb="select"]:focus-within > div {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-tint);
}

/* file uploader dropzone — dashed warm border */
[data-testid="stFileUploaderDropzone"] {
    border: 1px dashed var(--line-2) !important;
    background: rgb(246, 243, 241) !important;
    border-radius: 10px !important;
    padding: 38px 24px !important;
    transition: background 0.12s ease, border-color 0.12s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent) !important;
    background: var(--accent-tint) !important;
}

/* primary 버튼 — 라운드 pill 테라코타 */
.stButton > button[kind="primary"] {
    background: var(--accent);
    color: #fff;
    border: 1px solid var(--accent);
    border-radius: 999px;
    padding: 11px 22px;
    font-size: 14px; font-weight: 500;
    transition: background 0.15s ease, border-color 0.15s ease, transform 0.08s ease;
}
.stButton > button[kind="primary"]:hover:not(:disabled) {
    background: var(--accent-strong);
    border-color: var(--accent-strong);
    transform: none;
    box-shadow: none;
}
.stButton > button[kind="primary"]:active:not(:disabled) { transform: translateY(1px); }
.stButton > button[kind="primary"]:disabled {
    background: transparent;
    color: var(--ink-4);
    border-color: var(--line-2);
    cursor: not-allowed;
}

/* 일반 secondary 버튼 — 메인 콘텐츠용 */
[data-testid="stAppViewBlockContainer"] .stButton > button {
    border-radius: 6px;
    border: 1px solid var(--line-2);
    background: var(--surface);
    color: var(--ink);
    font-weight: 500;
    transition: border-color 0.12s ease, background 0.12s ease;
    padding: 0.5rem 1rem;
}
[data-testid="stAppViewBlockContainer"] .stButton > button:hover:not(:disabled) {
    border-color: var(--ink-3);
    background: var(--surface-tint);
    transform: none;
    box-shadow: none;
}

/* 카드 */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px;
    border-color: var(--line);
    background: var(--surface);
    transition: border-color 0.15s ease;
}

/* metric */
[data-testid="stMetricValue"] {
    font-weight: 600; letter-spacing: -0.02em; color: var(--ink);
}
[data-testid="stMetricLabel"] {
    color: var(--ink-3); font-weight: 500;
}

/* expander */
[data-testid="stExpander"] summary {
    transition: background 0.12s ease; border-radius: 8px;
}
[data-testid="stExpander"] summary:hover { background: var(--surface-tint); }

/* 영상 플레이어 */
[data-testid="stVideo"] video { border-radius: 10px; }

/* divider */
hr { border-color: var(--line); }

/* H1 (Streamlit기본) — 메인 영역에서만 사용 시 sans 그대로 */
[data-testid="stMarkdownContainer"] h1 { font-weight: 700; letter-spacing: -0.025em; }
[data-testid="stMarkdownContainer"] h2 { font-weight: 600; letter-spacing: -0.018em; }
</style>
        """,
        unsafe_allow_html=True,
    )


_inject_global_styles()

# 세션 상태 초기화
for _k, _v in (
    ("selected_analysis_id", None),
    ("is_analyzing", False),
    ("pending_upload", None),
    ("pending_category", None),
):
    if _k not in st.session_state:
        st.session_state[_k] = _v

is_analyzing: bool = st.session_state["is_analyzing"]

# 사이드바 — Claude warm editorial brand mark + "이전 기록"
with st.sidebar:
    st.markdown(
        '<div class="vid-brand">'
        '<div class="vid-brand-mark">V</div>'
        '<div class="vid-brand-name">vidoctor</div>'
        "</div>"
        '<div class="vid-sidebar-title">이전 기록</div>',
        unsafe_allow_html=True,
    )
    for a in _cached_list_analyses(limit=20):
        if st.button(
            _format_analysis_label(a),
            key=f"sel-{a['id']}",
            use_container_width=True,
            disabled=is_analyzing,
        ):
            st.session_state["selected_analysis_id"] = a["id"]
            st.rerun()


# 메인 — 3-state: analyzing / result / idle
selected_id = st.session_state.get("selected_analysis_id")

if is_analyzing:
    pending_file = st.session_state.get("pending_upload")
    pending_category = cast(Category, st.session_state.get("pending_category"))

    st.markdown(
        '<h1 class="vid-h1"><span class="vid-accent">분석</span> 진행 중</h1>'
        '<p class="vid-lede">영상 업로드부터 5차원 검출까지 — 진행 상황을 그래프로 보여드립니다.</p>',
        unsafe_allow_html=True,
    )

    graph_box = st.empty()
    upload_done = [False]
    completed: set[str] = set()

    def _rerender() -> None:
        graph_box.markdown(
            _render_dimension_graph(completed, pending_category, upload_done[0]),
            unsafe_allow_html=True,
        )

    def _on_upload() -> None:
        upload_done[0] = True
        _rerender()

    def _on_node(name: str) -> None:
        completed.add(name)
        _rerender()

    _rerender()

    try:
        analysis_id = _run_analysis_pipeline(
            pending_file, pending_category, _on_upload, _on_node
        )
        st.session_state["selected_analysis_id"] = analysis_id
        _invalidate_caches()
    except Exception as e:  # noqa: BLE001 — 사용자에게 표면화하면 충분
        st.error(f"분석 실패: {e}")
    finally:
        st.session_state["is_analyzing"] = False
        st.session_state["pending_upload"] = None
        st.session_state["pending_category"] = None
        st.rerun()

elif selected_id:
    st.markdown(
        '<h1 class="vid-h1"><span class="vid-accent">분석</span> 결과</h1>',
        unsafe_allow_html=True,
    )
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
    # idle — Claude warm editorial 헤딩 + 번호 필드 + pill submit
    st.markdown(
        '<h1 class="vid-h1"><span class="vid-accent">분석</span> 시작하기</h1>'
        '<p class="vid-lede">'
        "영상 카테고리를 선택하고 파일을 업로드하면 자동으로 분석을 시작합니다.<br/>"
        "업로드 즉시 처리되며, 결과는 좌측 <b>이전 기록</b>에 저장됩니다."
        "</p>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(
            '<div class="vid-field-head">'
            '<span class="vid-field-num">01</span>'
            '<span class="vid-field-label">카테고리</span>'
            '<span class="vid-field-hint">required</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        category = cast(
            Category,
            st.selectbox(
                "카테고리",
                options=list(CATEGORY_LABEL.keys()),
                format_func=lambda x: CATEGORY_LABEL[x],
                label_visibility="collapsed",
            ),
        )

        st.markdown(
            '<div class="vid-field-head">'
            '<span class="vid-field-num">02</span>'
            '<span class="vid-field-label">영상 업로드</span>'
            '<span class="vid-field-hint">max 300MB</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "영상 파일",
            type=["mp4", "mov"],
            accept_multiple_files=False,
            label_visibility="collapsed",
        )

        col_btn, col_hint = st.columns([1, 3], vertical_alignment="center")
        with col_btn:
            start_clicked = st.button(
                "분석 시작 →",
                disabled=uploaded is None,
                type="primary",
                use_container_width=True,
            )
        with col_hint:
            if uploaded is None:
                st.markdown(
                    '<span class="vid-submit-hint">파일을 업로드하면 활성화됩니다.</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="vid-submit-hint">'
                    '<span class="ok">●</span> 준비 완료 — 분석을 시작할 수 있습니다.'
                    "</span>",
                    unsafe_allow_html=True,
                )

    if start_clicked:
        st.session_state["pending_upload"] = uploaded
        st.session_state["pending_category"] = category
        st.session_state["is_analyzing"] = True
        st.rerun()
