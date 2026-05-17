"""5차원 분석 graph의 state·도메인 타입 정의 — TypedDict + Pydantic Event 모델."""

from collections.abc import Iterator
from operator import add
from typing import Annotated, Any, Literal, NotRequired, TypedDict, cast

import numpy as np
from pydantic import BaseModel, Field

from vidoctor.llm import LLMCallMetrics

Category = Literal["lecture", "vlog", "other"]
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

# 카테고리별 활성 차원. LangGraph conditional edge가 매핑 보고 그래프 자체 분기.
# vlog는 시선 이탈·내용 공백 비활성 (의도된 컷어웨이·일상 영상),
# other는 시선 이탈만 비활성 (음악·게임·예능 등 도메인 의존성 큼).
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


class FillerEvent(BaseModel):
    """필러(어, 음, 그) 발화 구간."""

    start: float
    end: float
    text: str

    def summary(self) -> str:
        return f"[{self.start:.1f}s] '{self.text}'"


class CPSEvent(BaseModel):
    """초당 글자 수 이상치 구간 (너무 빠르거나 느림)."""

    start: float
    end: float
    cps: float
    kind: Literal["too_fast", "too_slow"]

    def summary(self) -> str:
        return f"[{self.start:.1f}–{self.end:.1f}s] {self.kind} (cps={self.cps:.2f})"


class DeadZoneEvent(BaseModel):
    """무발화 구간 (긴 침묵)."""

    start: float
    end: float

    def summary(self) -> str:
        return f"[{self.start:.1f}–{self.end:.1f}s] 무발화 {self.end - self.start:.1f}s"


class GazeEvent(BaseModel):
    """시선 이탈 구간 (방향 포함)."""

    start: float
    end: float
    direction: Direction

    def summary(self) -> str:
        return f"[{self.start:.1f}–{self.end:.1f}s] 시선 이탈 {self.direction}"


class ContentGapEvent(BaseModel):
    """내용 공백 구간 (LLM이 발견)."""

    start: float
    end: float
    description: str

    def summary(self) -> str:
        return f"[{self.start:.1f}–{self.end:.1f}s] {self.description}"


class Suggestion(BaseModel):
    text: str
    finding_refs: list[str] = Field(default_factory=list)


class AnalysisState(TypedDict):
    """LangGraph 5차원 분석 state.

    video_path / category는 진입 시점에 필수, 나머지는 노드가 차례로 채움.
    각 detection 노드가 자기 필드만 채우므로 reducer 불필요 (default replace).
    """

    video_path: str
    category: Category

    transcript: NotRequired[list[Word]]

    # WhisperX가 디코딩한 16kHz mono 신호를 dead_zone(VAD) 재사용. transcribe 한 번이면
    # ffmpeg를 또 부르지 않는다. ndarray라 LangGraph checkpoint 직렬화 안 함 — 메모리만.
    audio_16k: NotRequired[np.ndarray]

    fillers: NotRequired[list[FillerEvent]]
    cps_anomalies: NotRequired[list[CPSEvent]]
    dead_zones: NotRequired[list[DeadZoneEvent]]
    gaze_issues: NotRequired[list[GazeEvent]]
    content_gaps: NotRequired[list[ContentGapEvent]]

    suggestions: NotRequired[list[Suggestion]]

    # LLM 호출 비용·latency 누적. operator.add reducer로 여러 LLM 노드가 각자 list를
    # 반환해도 LangGraph가 자동 concat. 영상당 총 비용은 합산해 산출.
    step_metrics: NotRequired[Annotated[list[LLMCallMetrics], add]]


def iter_dimension_events(
    state: AnalysisState,
) -> Iterator[tuple[Dimension, list[Any]]]:
    """차원 → event 리스트를 순회. 비활성 차원은 빈 리스트로 yield.

    TypedDict 동적 키 접근의 `# type: ignore[literal-required]`를 한 곳에 모은다.
    """
    for dim, field_name in DIM_TO_STATE_FIELD.items():
        events = state.get(field_name, []) or []  # type: ignore[literal-required]
        yield dim, list(events)


# Suggestion.finding_refs 직렬화. 형식 변경 시 이 한 쌍의 함수만 수정.
def format_finding_ref(dimension: Dimension, idx: int) -> str:
    return f"{dimension}:{idx}"


def parse_finding_ref(ref: str) -> tuple[Dimension, int] | None:
    """ref → (dimension, idx). 형식이 깨졌거나 dimension이 알 수 없으면 None."""
    if ":" not in ref:
        return None
    dim_str, idx_str = ref.split(":", 1)
    if not idx_str.isdigit() or dim_str not in DIM_TO_STATE_FIELD:
        return None
    return cast(Dimension, dim_str), int(idx_str)
