"""5차원 finding 종합 → 개선 제안 생성.

각 차원의 event를 LLM(gpt-4o-mini)에 전달, 한국어 제안을 받는다. 카테고리별 hardcoded
rule 없이 차원 신호의 일반적 의미만 정의 — 영상 도메인은 finding 패턴에서 LLM이 추정.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from vidoctor.graph.state import (
    CATEGORY_DIMENSIONS,
    AnalysisState,
    Dimension,
    Suggestion,
    Word,
    format_finding_ref,
    iter_dimension_events,
)
from vidoctor.llm import (
    LLMCallMetrics,
    get_chat_model,
    invoke_structured_with_metrics,
)

_log = logging.getLogger(__name__)

_MODEL = "gpt-4o-mini"

MAX_SUGGESTIONS = 8

# 차원당 LLM에 전달할 finding 상한. 그 너머는 "and N more"로 집계해 prompt 폭발 방지.
# finding이 폭증해도 LLM은 패턴만 인식하면 되므로 상위 N건이면 충분.
MAX_FINDINGS_PER_DIM = 30

# transcript 전체를 LLM에 그대로 넘겨 영상 주제·맥락을 인식시킨다. 5분 영상 한국어
# 기준 ~6K char(약 8K token). 그 이상은 머리·꼬리만 남기고 중간 생략 — 영상 도입부와
# 마무리에 주제 신호가 가장 강하다는 가정.
TRANSCRIPT_CHAR_CAP = 12000

# transcript를 30초 단위 chunk로 묶어 [Ns~] 헤딩을 박음. finding 시점과 매칭이 쉬워지고
# LLM이 "어느 구간이 어떤 주제였는지" 추론 가능.
TRANSCRIPT_CHUNK_SEC = 30.0

_RUBRIC = """당신은 영상 감수 전문가입니다. 아래 영상의 음성 전사(transcript)와 \
5차원 분석 finding을 함께 검토하고, 시청자 경험 개선을 위한 구체적·실행 가능한 \
제안을 작성하세요.

각 차원의 신호와 그에 맞는 행동 영역 (다른 차원의 행동을 권하지 말 것):
- filler: 군더더기 어휘 → 다음 녹화에서 의식적 줄이기 / 편집으로 컷 / 호흡 정리
- cps too_fast: 발화 속도 너무 빠름 → 호흡 추가 / 문장 단위 끊기 / 핵심에서 속도 늦추기
- cps too_slow: 발화 속도 너무 느림 → 군더더기 제거 / 문장 압축 / 주제 전개 가속화
  (cps 신호로 콘텐츠 추가·예시 보강 같은 콘텐츠 행동을 권하지 말 것 — 그것은 \
content_gap·dead_zone의 영역)
- dead_zone: 무발화·정적 화면 → 추가 설명·예시 삽입 / 시각 자료 추가 / 컷 편집
- gaze: 화자 응시 깨짐 → 카메라 위치 조정 / 시선 처리 연습 / 프레임 안정화
- content_gap: 화면 시각 ↔ 발화 불일치 → 슬라이드 수정 / 발화 수정 / 별도 자료 보강·도식 추가

작성 원칙:
- **finding_refs에 적은 finding 신호의 본질을 그 제안이 직접 다루어야 한다.** \
transcript는 톤·예시·시점 표현에만 활용하고, finding과 무관한 콘텐츠 개선 제안은 \
만들지 말 것 (예: cps finding 근거로 "예시를 추가하세요" 같은 콘텐츠 보강 제안 금지).
- transcript에서 영상 주제·핵심 개념을 파악해 제안 본문에 **영상에서 실제로 다룬 \
주제·용어**를 반영하라. 일반론("발화를 정리하세요")은 피하고, 그 영상에서 다루는 \
실제 단계·재료·악구·구문·도구 이름을 인용해 무엇을 어떻게 보강·정리할지 짚어라.
- 같은 차원·근접 시간대 finding은 묶어 한 제안으로 요약하라. \
finding 1건마다 제안 1건씩 나누지 말 것.
- 단, **cps는 kind별로 분리**해 묶어라 — too_fast finding들은 한 제안으로, \
too_slow finding들은 또 다른 제안으로. 두 kind는 행동(호흡·끊기 vs 압축·가속화)이 \
정반대라 한 제안에 섞으면 안 된다.
- 각 제안의 finding_refs에 근거 finding의 ref(예: "filler:0", "cps:2")를 적어라. \
**본문(text)에는 시간 표기를 넣지 말 것** — 시간은 UI가 finding_refs 버튼으로 \
별도 표시한다. 본문에 "[58s~]" "[00:01:12]" "47초 부근" 같은 시간 명시 금지.
- 영상 도메인(강의·브이로그·기타)은 transcript와 finding 패턴에서 추정해 톤을 조정하라. \
카테고리 가정 없이도 구체적 제안이 가능해야 한다.
- 미스매치·문제가 거의 없으면 빈 리스트.
"""


@dataclass(frozen=True)
class _Finding:
    """LLM 프롬프트에 넣는 finding 한 건 — ref 식별자 + 차원 + 축약 텍스트."""

    ref: str
    dimension: Dimension
    summary: str


@dataclass(frozen=True)
class _CollectedFindings:
    """LLM에 전달할 finding 묶음 + 차원당 cap 초과 카운트."""

    findings: list[_Finding]
    extras: dict[Dimension, int]  # 차원당 cap을 넘어 누락된 finding 수


class _SuggestionItem(BaseModel):
    text: str = Field(
        description=(
            "제안 본문, 한국어 1~2문장 행동 지시형. 영상 주제·구간 내용을 반영해 "
            "구체적으로."
        ),
        max_length=400,
    )
    finding_refs: list[str] = Field(
        default_factory=list,
        description="근거 finding 인덱스 (예: ['filler:0', 'cps:2'])",
    )


class _SuggestionResponse(BaseModel):
    suggestions: list[_SuggestionItem] = Field(
        default_factory=list, max_length=MAX_SUGGESTIONS
    )


def _format_transcript(transcript: list[Word]) -> str:
    """transcript를 30초 단위 chunk로 묶어 시점 헤딩이 박힌 텍스트로.

    cap 초과 시 머리·꼬리만 보존하고 중간을 생략 — 도입부·마무리에 영상 주제 신호가
    가장 강하다는 가정. finding은 별도 ref·시점으로 LLM에 전달되므로 중간 누락 정보는
    finding 메타로 메꿔진다.
    """
    if not transcript:
        return ""
    chunks: dict[int, list[str]] = {}
    for w in transcript:
        bucket = int(w.start // TRANSCRIPT_CHUNK_SEC) * int(TRANSCRIPT_CHUNK_SEC)
        chunks.setdefault(bucket, []).append(w.text)
    blocks = [f"[{start}s~] {' '.join(words)}" for start, words in sorted(chunks.items())]
    text = "\n".join(blocks)
    if len(text) <= TRANSCRIPT_CHAR_CAP:
        return text
    half = TRANSCRIPT_CHAR_CAP // 2 - 30
    # chunk 경계(\n)에서 잘라 발화 중간 단절을 피한다.
    head_cut = text.rfind("\n", 0, half)
    head = text[: head_cut if head_cut > 0 else half]
    tail_cut = text.find("\n", len(text) - half)
    tail = text[tail_cut + 1 if tail_cut > 0 else -half :]
    return head + "\n... (중간 생략) ...\n" + tail


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
                _Finding(ref=format_finding_ref(dim, i), dimension=dim, summary=e.summary())
            )
        if len(events) > MAX_FINDINGS_PER_DIM:
            extras[dim] = len(events) - MAX_FINDINGS_PER_DIM
    return _CollectedFindings(findings=findings, extras=extras)


@dataclass(frozen=True)
class _RefValidationStats:
    """LLM 출력 finding_refs 검증 결과 통계 (모니터링·테스트용)."""

    input_suggestions: int
    kept_suggestions: int
    invalid_refs_removed: int
    suggestions_dropped: int  # 유효 ref가 0개로 떨어져 통째로 drop된 suggestion 수


def _validate_refs(
    items: list[_SuggestionItem], valid_refs: set[str]
) -> tuple[list[_SuggestionItem], _RefValidationStats]:
    """LLM이 환각한 ref를 제거하고, 근거가 0이 된 suggestion은 drop.

    LLM은 종종 prompt에 없던 ref(예: 'filler:99', 'gaze:0' 미존재)를 만들어내거나
    오타를 낸다. UI는 invalid ref를 '(?)'로 표시할 수 있지만 — 그 단계에 도달하기 전에
    저장 직전에 걸러내는 것이 사용자 신뢰·DB 정합성 측면에서 낫다. 검증 결과는 stats로
    노출해 호출부가 LLM 품질을 추적할 수 있게 한다.
    """
    cleaned: list[_SuggestionItem] = []
    invalid_count = 0
    dropped = 0
    for item in items:
        valid = [r for r in item.finding_refs if r in valid_refs]
        invalid_count += len(item.finding_refs) - len(valid)
        if not valid:
            dropped += 1
            continue
        cleaned.append(item.model_copy(update={"finding_refs": valid}))
    return cleaned, _RefValidationStats(
        input_suggestions=len(items),
        kept_suggestions=len(cleaned),
        invalid_refs_removed=invalid_count,
        suggestions_dropped=dropped,
    )


def _build_message(collected: _CollectedFindings, state: AnalysisState) -> HumanMessage:
    """rubric + 카테고리 + 음성 전사 + finding 묶음을 LLM 메시지로 결합."""
    category = state["category"]
    active = ", ".join(CATEGORY_DIMENSIONS[category])
    transcript_text = _format_transcript(state.get("transcript", []) or [])

    blocks: list[str] = [
        _RUBRIC,
        f"영상 카테고리(참고용): {category}",
        f"활성 차원: {active}",
    ]
    if transcript_text:
        blocks.append("음성 전사 (시점 헤딩 박힘):\n" + transcript_text)
    else:
        blocks.append("음성 전사 없음 — finding만으로 제안 작성.")

    if not collected.findings:
        blocks.append("발견된 finding 없음.")
    else:
        blocks.append("Findings:")
        for f in collected.findings:
            blocks.append(f"- {f.ref} | {f.summary}")
        for dim, more in collected.extras.items():
            blocks.append(f"- ({dim} 차원 추가 {more}건 생략)")
    blocks.append(
        f"transcript에서 영상 주제·핵심 내용을 먼저 파악한 뒤, finding을 근거로 "
        f"개선 제안을 최대 {MAX_SUGGESTIONS}건 작성하라. 근거가 약하거나 신호가 "
        "미미하면 제안하지 말 것."
    )
    return HumanMessage(content="\n".join(blocks))


async def build_suggestions(
    state: AnalysisState,
) -> tuple[list[Suggestion], LLMCallMetrics]:
    """AnalysisState → (개선 제안 리스트, LLM 호출 메타).

    finding 0건이면 LLM 호출 생략. 도메인 가정 없이 차원 신호만 전달해 카테고리별
    오버핏을 회피한다.
    """
    collected = _collect_findings(state)
    if not collected.findings:
        return [], LLMCallMetrics.empty("suggestions", _MODEL)

    message = _build_message(collected, state)
    # max_tokens=1024: 8 suggestion × 1~2문장(~120 token) + finding_refs JSON 여유.
    # temperature=0.3: 같은 finding에 매번 동일 표현만 나오지 않게 약간의 변동 허용.
    model = get_chat_model(model=_MODEL, temperature=0.3, max_tokens=1024)
    parsed, metrics = await invoke_structured_with_metrics(
        model, _SuggestionResponse, [message], step="suggestions"
    )

    valid_refs = {f.ref for f in collected.findings}
    cleaned, stats = _validate_refs(parsed.suggestions, valid_refs)
    if stats.invalid_refs_removed or stats.suggestions_dropped:
        _log.info(
            "제안 ref 검증: 유지=%d/%d 환각=%d 드롭=%d",
            stats.kept_suggestions,
            stats.input_suggestions,
            stats.invalid_refs_removed,
            stats.suggestions_dropped,
        )

    suggestions = [
        Suggestion(text=s.text, finding_refs=s.finding_refs)
        for s in cleaned
    ]
    return suggestions, metrics
