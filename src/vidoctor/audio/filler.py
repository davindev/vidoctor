"""한국어 filler 검출 — 사전 매칭 + 인접 반복 묶음.

Shriberg(1994) disfluency 4분류(filled pause/repetition/repair/restart)에서 분류 체계 영향.
사전 외 단어 반복은 무시 — 강조·명령("짜잔 짜잔") 케이스가 disfluency repetition 가정보다
우세해 vlog 검증에서 가정 깨짐.
"""

from __future__ import annotations

import re

from vidoctor.graph.state import FillerEvent, Word

# 한국어 구어에서 자주 보고되는 filled pause 어휘. 명확형(어/음/...)과 모호형(이제/막/...) 통합.
# "그러니까/그래서"는 의도적 미포함 (강의 논리 연결사로 정상 사용 → FP 다수).
# "자"는 주의 환기 표지("자, 이제…")로 포함.
FILLERS: frozenset[str] = frozenset(
    {
        # 명확형 단음절·지시사·환기 표지.
        "어", "음", "으", "에",
        "그", "저",
        "자",
        "뭐", "뭐지", "뭐랄까",
        # 모호형 (일반 단어로도 쓰임 — 머뭇거림 의도일 때만 의미)
        "이제", "인제",
        "막", "좀", "약간",
    }
)

# 같은 어휘 반복 묶음 임계. 정상 단어 간 휴지 50~200ms vs Shriberg(1994) editing region
# (머뭇거림성 반복·수정 사이) 평균 300~700ms — 중간점 500ms 채택.
REPETITION_GAP_THRESHOLD_SEC = 0.5

# 어휘 무관 burst chain 임계 — pyannote/silero VAD의 min_silence_duration 컨벤션.
# "한 머뭇거림 상태 = 1 알림" UX 의도. lecture 골든셋 burst 안 최대 gap(4s)을 흡수하고
# 라벨 간 거리(20s+)와 충분히 분리되는 5s.
BURST_MERGE_GAP_SEC = 5.0

# burst 묶음 길이 cap. 60초 넘는 연속 머뭇거림은 "비정상 산만함"이라 단일 알림이
# 오히려 심각도를 가리므로 강제 분할해 여러 finding으로 노출. silero VAD의
# max_speech_duration_s 30s보다 약간 길게 — filler burst는 일반 speech보다 길게 이어짐.
MAX_BURST_DURATION_SEC = 60.0

_PUNCT_RE = re.compile(r"[^\w가-힣]")


def normalize_word(text: str) -> str:
    """단어에서 한글·영숫자 외 문자(구두점 등)를 제거한 정규화 결과."""
    return _PUNCT_RE.sub("", text).strip()


def detect_filler_events(
    words: list[Word], burst_gap: float | None = BURST_MERGE_GAP_SEC
) -> list[FillerEvent]:
    """단어 시퀀스에서 filler 후보 추출.

    1단계: 같은 어휘 반복(run)은 REPETITION_GAP_THRESHOLD_SEC 이내면 단일 이벤트로.
    2단계: burst_gap이 주어지면 어휘 무관으로 인접 events를 합쳐 "한 머뭇거림 burst =
    알림 1건" UX 의도를 구현. None이면 1단계만(legacy 동작).

    사전 외 단어 반복은 무시 — vlog 검증 결과 "인접 반복 = disfluency" 가정이
    강조/명령(예: "짜잔 짜잔") 케이스에 뒤집힘. 강의도 강조용 반복이 자연스러워 동일 정책.
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

    return _merge_burst(events, burst_gap) if burst_gap is not None else events


def _merge_burst(events: list[FillerEvent], gap: float) -> list[FillerEvent]:
    """1단계 events 위에 어휘 무관 burst 묶음. 인접 event 간 gap 이내면 합치되,
    묶음 길이가 MAX_BURST_DURATION_SEC 넘으면 새 묶음으로 분할."""
    if not events:
        return events
    merged = [events[0]]
    for ev in events[1:]:
        head = merged[-1]
        within_gap = ev.start - head.end < gap
        within_cap = ev.end - head.start <= MAX_BURST_DURATION_SEC
        if within_gap and within_cap:
            merged[-1] = FillerEvent(
                start=head.start,
                end=ev.end,
                text=f"{head.text} {ev.text}",
            )
        else:
            merged.append(ev)
    return merged
