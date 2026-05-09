"""한국어 filler 검출.

분류 체계는 Shriberg(1994) disfluency 4분류(filled pause/repetition/repair/restart)에서
영향을 받음. 사전은 한국어 구어에서 자주 보고되는 filler 어휘 기반
(정밀 검증·튜닝은 골든셋 라벨링 후).

검출 정책 (사전 매칭 단일 차원):
- 사전 단어(어/음/으/에/그/저/뭐, 이제/막/좀/약간 등)만 filler로 등록
- 인접 반복은 묶어서 단일 이벤트(예: "어 어 어" → 한 이벤트)
- 사전에 없는 단어의 반복은 무시 — 강조/명령("짜잔 짜잔", "강아지 강아지")이
  머뭇거림보다 우세해 disfluency repetition 가정이 데이터에서 깨짐(vlog 검증)

self-correction / backchannel 차원 확장 로드맵은 README 참조.
"""

from __future__ import annotations

import re

from vidoctor.graph.state import FillerEvent, Word

# 한국어 구어에서 자주 보고되는 filled pause 어휘. 명확형(어/음/...)과 모호형(이제/막/...)을
# 단일 사전으로 묶었다. 어휘 차등은 평가 기준이 갖춰진 뒤 다시 분리 — 그때까진 git history에
# 출처가 보존되니 미래 분기를 위해 자료구조를 분리해 둘 필요 없다.
#
# "그러니까/그래서"는 의도적으로 미포함: 강의에서 논리 연결사로 정상 사용되는 비율이 높아
# false positive 다수. "자"는 주의 환기 표지("자, 이제…")로 한국어 구어에서 자주 쓰여 포함.
FILLERS: frozenset[str] = frozenset(
    {
        # 명확형 단음절·지시사·환기 표지. 발화 길이가 길수록 ASR 토큰화 안정 —
        # lecture의 의도적 늘임("음...")은 잡히고, vlog 짧은 burst(0.1~0.3s)는
        # 정규화 흡수되어 detection input 부재. 후자는 v1.1 자체 fine-tune 영역.
        "어", "음", "으", "에",
        "그", "저",
        "자",
        "뭐", "뭐지", "뭐랄까",
        # 모호형 (일반 단어로도 쓰임 — 머뭇거림 의도일 때만 의미)
        "이제", "인제",
        "막", "좀", "약간",
    }
)

# 반복 인접성 기준 (휴리스틱). 정상 단어 간 휴지는 50~200ms,
# Shriberg(1994)의 editing region(머뭇거림성 반복·수정 사이)은 평균 300~700ms.
# 정상 휴지를 명확히 넘으면서 editing region 중간점인 500ms 채택.
# 골든셋의 인접 동일어 시간차 분포로 갱신 예정.
REPETITION_GAP_THRESHOLD_SEC = 0.5

_PUNCT_RE = re.compile(r"[^\w가-힣]")


def normalize_word(text: str) -> str:
    return _PUNCT_RE.sub("", text).strip()


def detect_filler_events(words: list[Word]) -> list[FillerEvent]:
    """단어 시퀀스에서 filler 후보 추출.

    사전 단어만 등록. 인접 반복(run)이면 묶어서 단일 이벤트.
    사전에 없는 단어 반복은 무시 — vlog 검증 결과 "인접 반복 = disfluency"
    가정이 강조/명령(예: 강아지 이름 호출, "짜잔 짜잔") 케이스에 뒤집힘.
    강의에서도 강조용 반복이 자연스러워 두 카테고리 모두 동일 정책.
    """
    normed = [(w, normalize_word(w.text)) for w in words]
    events: list[FillerEvent] = []

    i = 0
    while i < len(words):
        _, norm = normed[i]
        if not norm or norm not in FILLERS:
            i += 1
            continue

        run_end = i + 1
        # 같은 어휘가 임계 이내 인접 → 한 머뭇거림 burst로 묶어 단일 finding으로 등록.
        # "한 번의 머뭇거림 = 사용자에게 알림 1건" UX 의도.
        while (
            run_end < len(words)
            and normed[run_end][1] == norm
            and (words[run_end].start - words[run_end - 1].end) < REPETITION_GAP_THRESHOLD_SEC
        ):
            run_end += 1

        events.append(
            FillerEvent(
                start=words[i].start,
                end=words[run_end - 1].end,
                text=" ".join(w.text for w in words[i:run_end]),
            )
        )
        i = run_end

    return events
