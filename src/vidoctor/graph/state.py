from typing import Literal, NotRequired, TypedDict

from pydantic import BaseModel

Category = Literal["lecture", "vlog", "other"]
Severity = Literal["low", "mid", "high"]
Dimension = Literal["filler", "cps", "dead_zone", "gaze", "content_gap"]
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

# 차원 → AnalysisState 필드 매핑. detector 출력 / DB 저장 / 평가 메트릭 모두 공유.
DIM_TO_STATE_FIELD: dict[Dimension, str] = {
    "filler": "fillers",
    "cps": "cps_anomalies",
    "dead_zone": "dead_zones",
    "gaze": "gaze_issues",
    "content_gap": "content_gaps",
}

# 카테고리별 활성 차원. LangGraph conditional edge가 이 매핑을 보고 그래프 자체를 분기 →
# 비활성 차원은 detection 노드가 호출되지 않는다.
# - lecture: 5차원 모두
# - vlog/인터뷰: 시선 이탈·내용 공백은 비활성 (의도된 컷어웨이·일상 기록 영상에 부적합)
# - other(default): 시선 이탈만 비활성 (도메인 의존성 큼 — 음악·게임·예능 등)
CATEGORY_DIMENSIONS: dict[Category, tuple[Dimension, ...]] = {
    "lecture": ("filler", "cps", "dead_zone", "gaze", "content_gap"),
    "vlog": ("filler", "cps", "dead_zone"),
    "other": ("filler", "cps", "dead_zone", "content_gap"),
}


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
