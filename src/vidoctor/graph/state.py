from typing import Literal, NotRequired, TypedDict

from pydantic import BaseModel

Category = Literal["lecture", "vlog", "other"]
Severity = Literal["low", "mid", "high"]
Direction = Literal[
    "front",
    "left",
    "right",
    "up",
    "down",
    "left_up",
    "left_down",
    "right_up",
    "right_down",
]


class Word(BaseModel):
    """WhisperX word-level timestamp."""

    text: str
    start: float
    end: float
    score: float | None = None


# severity는 모든 차원에서 default "mid"로 통일. detector별 임계 결정 근거(라벨링·평가
# 시스템)가 갖춰지기 전엔 분기가 노이즈만 만들어 평가 정확도 저해. 로드맵은 README 참조.
class FillerEvent(BaseModel):
    start: float
    end: float
    text: str
    severity: Severity = "mid"


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
    direction: Direction
    severity: Severity = "mid"


class ContentGapEvent(BaseModel):
    start: float
    end: float
    description: str
    severity: Severity = "mid"


class Suggestion(BaseModel):
    text: str
    priority: int = 0
    finding_refs: list[str] = []


class AnalysisState(TypedDict):
    """LangGraph 5차원 분석 state.

    video_path / category는 진입 시점에 필수, 나머지는 노드가 차례로 채움.
    각 detection 노드가 자기 필드만 채우므로 reducer 불필요 (default replace).
    """

    video_path: str
    category: Category

    transcript: NotRequired[list[Word]]

    fillers: NotRequired[list[FillerEvent]]
    cps_anomalies: NotRequired[list[CPSEvent]]
    dead_zones: NotRequired[list[DeadZoneEvent]]
    gaze_issues: NotRequired[list[GazeEvent]]
    content_gaps: NotRequired[list[ContentGapEvent]]

    suggestions: NotRequired[list[Suggestion]]
