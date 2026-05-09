"""5차원 finding 종합 → 개선 제안 생성.

각 차원의 event를 LLM(gpt-4o-mini)에 전달, 한국어 제안을 받는다. 카테고리별 hardcoded
rule 없이 차원 신호의 일반적 의미만 정의 — 영상 도메인은 finding 패턴에서 LLM이 추정.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from vidoctor.graph.state import (
    CATEGORY_DIMENSIONS,
    AnalysisState,
    Dimension,
    Suggestion,
    iter_dimension_events,
)
from vidoctor.llm import get_chat_model

MAX_SUGGESTIONS = 8

# 차원당 LLM에 전달할 finding 상한. 그 너머는 "and N more"로 집계해 prompt 폭발 방지.
# finding이 폭증해도 LLM은 패턴만 인식하면 되므로 상위 N건이면 충분.
MAX_FINDINGS_PER_DIM = 30

_RUBRIC = """당신은 영상 감수 전문가입니다. 아래 영상의 5차원 분석 결과를 검토하고, \
시청자 경험 개선을 위한 구체적·실행 가능한 제안을 작성하세요.

각 차원이 가리키는 신호:
- filler: 군더더기 어휘로 전달력·자신감 약화
- cps: 발화 속도 이상(너무 빠름/느림) — 청취 부담
- dead_zone: 무발화·정적 화면이 길어 시청자 이탈 위험
- gaze: 화자 응시 안정성 (시선 이탈)
- content_gap: 화면 시각 정보와 발화 정보의 불일치

작성 원칙:
- 제안은 한국어 한 문장. 행동 지시형으로 (예: "X를 정리하세요").
- 같은 차원·근접 시간대 finding은 묶어 한 제안으로 요약하라. \
finding 1건마다 제안 1건씩 나누지 말 것.
- 각 제안의 finding_refs에 근거 finding의 ref(예: "filler:0", "cps:2")를 적어라.
- priority: 0이 가장 높음. 빈도·구간 길이·시청자 영향 기반으로 결정.
- 영상 도메인(강의·브이로그·기타)은 finding 패턴에서 추정해 톤을 조정하라. \
카테고리 가정 없이도 구체적 제안이 가능해야 한다.
- 미스매치·문제가 거의 없으면 빈 리스트.
"""


@dataclass(frozen=True)
class _Finding:
    ref: str
    dimension: Dimension
    summary: str


@dataclass(frozen=True)
class _CollectedFindings:
    findings: list[_Finding]
    extras: dict[Dimension, int]  # 차원당 cap을 넘어 누락된 finding 수


class _SuggestionItem(BaseModel):
    text: str = Field(
        description="제안 본문, 한국어 한 문장 행동 지시형",
        max_length=200,
    )
    priority: int = Field(description="우선순위 (0=가장 높음)", ge=0)
    finding_refs: list[str] = Field(
        default_factory=list,
        description="근거 finding 인덱스 (예: ['filler:0', 'cps:2'])",
    )


class _SuggestionResponse(BaseModel):
    suggestions: list[_SuggestionItem] = Field(
        default_factory=list, max_length=MAX_SUGGESTIONS
    )


def _collect_findings(state: AnalysisState) -> _CollectedFindings:
    """state의 모든 차원 event를 _Finding 리스트로. 차원당 MAX_FINDINGS_PER_DIM cap."""
    findings: list[_Finding] = []
    extras: dict[Dimension, int] = {}
    for dim, events in iter_dimension_events(state):
        if not events:
            continue
        capped = events[:MAX_FINDINGS_PER_DIM]
        for i, e in enumerate(capped):
            findings.append(
                _Finding(ref=f"{dim}:{i}", dimension=dim, summary=e.summary())
            )
        if len(events) > MAX_FINDINGS_PER_DIM:
            extras[dim] = len(events) - MAX_FINDINGS_PER_DIM
    return _CollectedFindings(findings=findings, extras=extras)


def _build_message(collected: _CollectedFindings, state: AnalysisState) -> HumanMessage:
    category = state["category"]
    active = ", ".join(CATEGORY_DIMENSIONS[category])
    blocks: list[str] = [
        _RUBRIC,
        f"영상 카테고리(참고용): {category}",
        f"활성 차원: {active}",
    ]
    if not collected.findings:
        blocks.append("발견된 finding 없음.")
    else:
        blocks.append("Findings:")
        for f in collected.findings:
            blocks.append(f"- {f.ref} | {f.summary}")
        for dim, more in collected.extras.items():
            blocks.append(f"- ({dim} 차원 추가 {more}건 생략)")
    blocks.append(
        f"위 finding을 근거로 개선 제안을 최대 {MAX_SUGGESTIONS}건 작성하라. "
        "근거가 약하거나 신호가 미미하면 제안하지 말 것."
    )
    return HumanMessage(content="\n".join(blocks))


async def build_suggestions(state: AnalysisState) -> list[Suggestion]:
    """AnalysisState → 개선 제안 리스트.

    finding 0건이면 LLM 호출 생략. 도메인 가정 없이 차원 신호만 전달해 카테고리별
    오버핏을 회피한다.
    """
    collected = _collect_findings(state)
    if not collected.findings:
        return []

    message = _build_message(collected, state)
    # max_tokens=512: 8건 × ~한 문장 + finding_refs JSON으로 충분.
    # temperature=0.3: 같은 finding에 매번 동일 표현만 나오지 않게 약간의 변동 허용.
    model = get_chat_model(model="gpt-4o-mini", temperature=0.3, max_tokens=512)
    structured = model.with_structured_output(_SuggestionResponse)
    response = cast(_SuggestionResponse, await structured.ainvoke([message]))

    return [
        Suggestion(text=s.text, priority=s.priority, finding_refs=s.finding_refs)
        for s in response.suggestions
    ]
