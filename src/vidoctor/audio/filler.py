"""한국어 filler 검출.

분류 체계는 Shriberg(1994) disfluency 4분류(filled pause/repetition/repair/restart)에서
영향을 받음. 사전은 한국어 구어에서 자주 보고되는 filler 어휘 기반
(정밀 검증·튜닝은 골든셋 라벨링 후).

MVP 검출 차원:
- Tier 1: 명확한 filled pause (어/음/으/에/그/저/뭐 ...)
- Tier 2: 모호한 filled pause (이제/막/좀/약간 — 일반 단어로도 쓰임)
- 모음 늘임: Tier 1 + duration ≥ 400ms (severity 상향)
- Repetition: 인접 동일어 + 시간차 < 500ms (run으로 묶어 단일 이벤트)

v1.1 도입 예정:
- Self-correction (repair + restart 통합): LLM 보조 휴리스틱 + 골든셋 측정 후
  정식 차원화. 두 발화 사이 의미 비교가 필요해 룰 기반 단독으론 정확도 한계.
- Backchannel: 화자 본인 발화와 응답 구분 어려움.
"""

from __future__ import annotations

import re

from vidoctor.graph.state import FillerEvent, Word

TIER_1_FILLERS: frozenset[str] = frozenset(
    {
        "어", "음", "으", "에",
        "그", "저",
        "뭐", "뭐지", "뭐랄까",
    }
)

TIER_2_FILLERS: frozenset[str] = frozenset(
    {
        "이제", "인제",
        "막", "좀", "약간",
        "그러니까", "그래서",
    }
)

# 모음 늘임 판정 기준 (휴리스틱). 일반 한국어 단음절은 150~250ms 범위가 보통이고
# Shriberg(1994) 영어 filled pause는 평균 300~500ms로 보고됨. 일반 단음절의 명확한
# 위쪽이면서 늘어진 발화 분포의 시작점에 해당하는 400ms 채택.
# 골든셋의 Tier 1 단어 duration 분포 측정 후 ROC F1 기준으로 갱신 예정.
LONG_VOWEL_THRESHOLD_SEC = 0.4

# 반복 인접성 기준 (휴리스틱). 정상 단어 간 휴지는 50~200ms,
# Shriberg(1994)의 editing region(머뭇거림성 반복·수정 사이)은 평균 300~700ms.
# 정상 휴지를 명확히 넘으면서 editing region 중간점인 500ms 채택.
# 골든셋의 인접 동일어 시간차 분포로 갱신 예정.
REPETITION_GAP_THRESHOLD_SEC = 0.5

_PUNCT_RE = re.compile(r"[^\w가-힣]")


def _normalize(text: str) -> str:
    return _PUNCT_RE.sub("", text).strip()


def _single_word_event(word: Word, norm: str) -> FillerEvent | None:
    if norm in TIER_1_FILLERS:
        duration = word.end - word.start
        severity = "mid" if duration >= LONG_VOWEL_THRESHOLD_SEC else "low"
        return FillerEvent(
            start=word.start, end=word.end, text=word.text, severity=severity
        )
    if norm in TIER_2_FILLERS:
        return FillerEvent(
            start=word.start, end=word.end, text=word.text, severity="low"
        )
    return None


def detect_filler_events(words: list[Word]) -> list[FillerEvent]:
    """단어 시퀀스에서 filler 후보 추출.

    인접 동일어 run을 먼저 그룹화 → 길이 1은 Tier 매칭, 길이 ≥ 2는 반복 이벤트로 단일 등록.
    이렇게 하면 같은 단어가 Tier 매칭 + 반복으로 이중 등록되는 사고 방지.
    """
    normed = [(w, _normalize(w.text)) for w in words]
    events: list[FillerEvent] = []

    i = 0
    while i < len(words):
        word, norm = normed[i]
        if not norm:
            i += 1
            continue

        run_end = i + 1
        while (
            run_end < len(words)
            and normed[run_end][1] == norm
            and (words[run_end].start - words[run_end - 1].end) < REPETITION_GAP_THRESHOLD_SEC
        ):
            run_end += 1

        if run_end - i >= 2:
            events.append(
                FillerEvent(
                    start=words[i].start,
                    end=words[run_end - 1].end,
                    text=" ".join(w.text for w in words[i:run_end]),
                    severity="mid",
                )
            )
        else:
            ev = _single_word_event(word, norm)
            if ev is not None:
                events.append(ev)

        i = run_end

    return events
