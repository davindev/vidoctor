"""Repository 변환 함수 단위 테스트.

순수 함수(_event_to_row / _row_to_event / _collect_finding_rows) 위주.
실 DB 통합 테스트는 별도 (Supabase 환경 필요).
"""

from __future__ import annotations

from pydantic import BaseModel

from vidoctor.graph.state import (
    AnalysisState,
    ContentGapEvent,
    CPSEvent,
    DeadZoneEvent,
    FillerEvent,
    GazeEvent,
)
from vidoctor.repository import (
    _collect_finding_rows,
    _event_to_row,
    _row_to_event,
)

# ---------------------------------------------------------------------------
# _event_to_row — 차원별 고유 필드는 payload로 분리되는지
# ---------------------------------------------------------------------------


def test_event_to_row_filler_payload_keeps_text():
    ev = FillerEvent(start=1.0, end=1.5, text="음")
    row = _event_to_row("aid-1", "filler", ev)
    assert row["analysis_id"] == "aid-1"
    assert row["dimension"] == "filler"
    assert row["start_sec"] == 1.0
    assert row["end_sec"] == 1.5
    assert row["payload"] == {"text": "음"}


def test_event_to_row_cps_payload_keeps_kind_and_value():
    ev = CPSEvent(start=2.0, end=12.0, cps=11.25, kind="too_fast")
    row = _event_to_row("aid", "cps", ev)
    assert row["payload"] == {"cps": 11.25, "kind": "too_fast"}


def test_event_to_row_dead_zone_payload_empty():
    ev = DeadZoneEvent(start=10.0, end=42.0)
    row = _event_to_row("aid", "dead_zone", ev)
    assert row["payload"] == {}


def test_event_to_row_gaze_payload_keeps_direction():
    ev = GazeEvent(start=5.0, end=8.0, direction="right_down")
    row = _event_to_row("aid", "gaze", ev)
    assert row["payload"] == {"direction": "right_down"}


def test_event_to_row_content_gap_payload_keeps_description():
    ev = ContentGapEvent(start=15.0, end=30.0, description="설명 부족")
    row = _event_to_row("aid", "content_gap", ev)
    assert row["payload"] == {"description": "설명 부족"}


# ---------------------------------------------------------------------------
# _row_to_event — DB row에서 정확한 이벤트 클래스로 복원
# ---------------------------------------------------------------------------


def test_row_to_event_returns_correct_class():
    row = {
        "dimension": "cps",
        "start_sec": 2.0,
        "end_sec": 12.0,
        "payload": {"cps": 11.25, "kind": "too_fast"},
    }
    ev = _row_to_event(row)
    assert isinstance(ev, CPSEvent)
    assert ev.cps == 11.25
    assert ev.kind == "too_fast"


def test_row_to_event_null_payload_treated_as_empty():
    row = {
        "dimension": "dead_zone",
        "start_sec": 10.0,
        "end_sec": 42.0,
        "payload": None,
    }
    ev = _row_to_event(row)
    assert isinstance(ev, DeadZoneEvent)


# ---------------------------------------------------------------------------
# 라운드트립 — 모든 차원이 손실 없이 복원되는지
# ---------------------------------------------------------------------------


def test_roundtrip_each_dimension():
    cases: list[tuple[str, BaseModel]] = [
        ("filler", FillerEvent(start=1.0, end=1.5, text="음")),
        (
            "cps",
            CPSEvent(start=2.0, end=12.0, cps=11.25, kind="too_fast"),
        ),
        ("dead_zone", DeadZoneEvent(start=10.0, end=42.0)),
        ("gaze", GazeEvent(start=5.0, end=8.0, direction="right_down")),
        (
            "content_gap",
            ContentGapEvent(start=15.0, end=30.0, description="설명 부족"),
        ),
    ]
    for dim, ev in cases:
        row = _event_to_row("aid", dim, ev)
        restored = _row_to_event(row)
        assert type(restored) is type(ev)
        assert restored.model_dump() == ev.model_dump()


# ---------------------------------------------------------------------------
# _collect_finding_rows — graph state → bulk insert rows
# ---------------------------------------------------------------------------


def test_collect_finding_rows_picks_only_present_dimensions():
    state: AnalysisState = {
        "video_path": "x",
        "category": "lecture",
        "fillers": [FillerEvent(start=0.0, end=0.5, text="음")],
        "cps_anomalies": [],
        "dead_zones": [DeadZoneEvent(start=10.0, end=42.0)],
        "gaze_issues": [GazeEvent(start=5.0, end=8.0, direction="right_down")],
        "content_gaps": [],
    }
    rows = _collect_finding_rows("aid", state)
    dims = sorted(r["dimension"] for r in rows)
    assert dims == ["dead_zone", "filler", "gaze"]


def test_collect_finding_rows_empty_state_returns_empty_list():
    state: AnalysisState = {"video_path": "x", "category": "lecture"}
    assert _collect_finding_rows("aid", state) == []


def test_collect_finding_rows_preserves_analysis_id_in_each_row():
    state: AnalysisState = {
        "video_path": "x",
        "category": "lecture",
        "fillers": [
            FillerEvent(start=0.0, end=0.5, text="음"),
            FillerEvent(start=2.0, end=2.5, text="어"),
        ],
    }
    rows = _collect_finding_rows("AID-42", state)
    assert all(r["analysis_id"] == "AID-42" for r in rows)
    assert len(rows) == 2
