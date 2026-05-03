"""한국어 filler 검출.

분류 체계는 Shriberg(1994) disfluency 4분류(filled pause/repetition/repair/restart)에서
영향을 받음. 사전은 한국어 구어에서 자주 보고되는 filler 어휘 기반
(정밀 검증·튜닝은 골든셋 라벨링 후).

MVP 검출 차원:
- Filled pause: 명확형(어/음/으/에/그/저/뭐) + 모호형(이제/막/좀/약간) 통합 사전
- Repetition: 인접 동일어 + 시간차 < 500ms (run으로 묶어 단일 이벤트)

severity는 default(mid). 임계 결정 근거(라벨링·평가)가 없는 상태에선 분기가 노이즈만
만든다. 로드맵(self-correction / backchannel / severity 차등)은 README 참조.
"""

from __future__ import annotations

import re

from vidoctor.graph.state import FillerEvent, Word

# 한국어 구어에서 자주 보고되는 filled pause 어휘. 명확형(어/음/...)과 모호형(이제/막/...)을
# 단일 사전으로 묶었다. 어휘 차등(severity 분기)은 평가 기준이 갖춰진 뒤 다시 분리 — 그때까진
# git history에 출처가 보존되니 미래 분기를 위해 자료구조를 분리해 둘 필요 없다.
FILLERS: frozenset[str] = frozenset(
    {
        # 명확형
        "어", "음", "으", "에",
        "그", "저",
        "뭐", "뭐지", "뭐랄까",
        # 모호형 (일반 단어로도 쓰임)
        "이제", "인제",
        "막", "좀", "약간",
        "그러니까", "그래서",
    }
)

# 반복 인접성 기준 (휴리스틱). 정상 단어 간 휴지는 50~200ms,
# Shriberg(1994)의 editing region(머뭇거림성 반복·수정 사이)은 평균 300~700ms.
# 정상 휴지를 명확히 넘으면서 editing region 중간점인 500ms 채택.
# 골든셋의 인접 동일어 시간차 분포로 갱신 예정.
REPETITION_GAP_THRESHOLD_SEC = 0.5

_PUNCT_RE = re.compile(r"[^\w가-힣]")


def _normalize(text: str) -> str:
    return _PUNCT_RE.sub("", text).strip()


def _single_word_event(word: Word, norm: str) -> FillerEvent | None:
    if norm in FILLERS:
        return FillerEvent(start=word.start, end=word.end, text=word.text)
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
                )
            )
        else:
            ev = _single_word_event(word, norm)
            if ev is not None:
                events.append(ev)

        i = run_end

    return events
