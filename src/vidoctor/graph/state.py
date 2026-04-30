from typing import Literal, TypedDict

from pydantic import BaseModel

Category = Literal["lecture", "vlog", "other"]
Severity = Literal["low", "mid", "high"]


class Word(BaseModel):
    """WhisperX word-level timestamp."""

    text: str
    start: float
    end: float
    score: float | None = None


class FillerEvent(BaseModel):
    start: float
    end: float
    text: str
    severity: Severity = "low"


class CPSEvent(BaseModel):
    start: float
    end: float
    cps: float
    kind: Literal["too_fast", "too_slow"]
    severity: Severity = "mid"


class DeadZoneEvent(BaseModel):
    start: float
    end: float
    severity: Severity = "mid"


class GazeEvent(BaseModel):
    start: float
    end: float
    direction: str
    severity: Severity = "low"


class ContentGapEvent(BaseModel):
    start: float
    end: float
    description: str
    severity: Severity = "mid"


class Suggestion(BaseModel):
    text: str
    priority: int = 0
    finding_refs: list[str] = []


class AnalysisState(TypedDict, total=False):
    """LangGraph 5차원 분석 state.

    각 detection 노드가 자기 필드만 채우므로 reducer 불필요 (default replace).
    """

    video_path: str
    category: Category

    transcript: list[Word]

    fillers: list[FillerEvent]
    cps_anomalies: list[CPSEvent]
    dead_zones: list[DeadZoneEvent]
    gaze_issues: list[GazeEvent]
    content_gaps: list[ContentGapEvent]

    suggestions: list[Suggestion]
