"""내용 공백 검출 — GPT-4o Vision multi-image batch + 카테고리 rubric.

브이로그는 비활성 (일상 기록·감정 공유 영상에 "설명 충분성" 평가 부적합).
강의는 "슬라이드 vs 발화 미스매치", 기타는 "자체 의미 전달 가능성" rubric.

알고리즘:
1. 영상에서 SAMPLE_INTERVAL_SEC(30s) 균등 + PySceneDetect 컷 경계 결합 프레임 시각 추출
2. dedup (인접 시각 5s 이내는 하나만) + MAX_SAMPLES 캡으로 비용 통제
3. 각 프레임을 720p 다운스케일 + JPEG 인코딩 + base64
4. 각 프레임 시각 ± WINDOW_SEC(15s) transcript 추출, 윈도우 외부에 단어 있으면 양 끝
   "…" 마커 부착 → rubric에서 "…는 끊김 아님" 안내. ASR 분할이 만드는 가짜 누락 신호 차단.
5. 프레임·transcript·rubric을 한 multi-image 메시지로 묶어 GPT-4o Vision 호출
6. Pydantic structured output(_ContentGapResponse)으로 파싱 → ContentGapEvent
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import cast

import cv2
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from scenedetect import ContentDetector, SceneManager, open_video

from vidoctor.graph.state import Category, ContentGapEvent, Word
from vidoctor.llm import get_chat_model

SAMPLE_INTERVAL_SEC = 30.0
TRANSCRIPT_WINDOW_SEC = 15.0
JPEG_QUALITY = 80
MAX_FRAME_HEIGHT = 720

# 컷 경계와 균등 샘플이 너무 가까우면 (이 시각 이내) 하나만 사용.
SCENE_DEDUP_THRESHOLD_SEC = 5.0

# 한 LLM 호출 최대 프레임 수. 비용·token 한계 고려.
# 3분 영상 균등 6장 + 컷 4~6개 + 캡 → 보통 8~10장.
MAX_SAMPLES = 10


_LECTURE_RUBRIC = """당신은 강의 영상 감수 전문가입니다. 아래 강의의 여러 시점 프레임과 \
해당 구간의 음성 전사를 검토하고, **슬라이드 화면이 가리키는 개념·정보와 강사 발화 \
내용이 어긋나는 구간**을 찾아내세요.

미스매치 판정 기준:
- 슬라이드 텍스트·도식이 표시한 용어·개념과 강사가 발화한 단어·정의가 다른 경우
- 슬라이드 항목과 발화 주제가 서로 다른 대상을 가리키는 경우

다음 신호는 미스매치가 아니므로 검출하지 마세요:
- 발화가 매끄럽지 않거나 짧은 추임새·반복이 있음
- 음성 전사가 윈도우 경계("…" 표기)에서 잘려 보이는 것 — 실제 발화는 인접 구간으로 연속
- 슬라이드와 발화가 같은 주제를 다루며 단지 표현·예시가 다른 경우

각 issue마다 mismatch_keyword 필드에 **강사가 발화한 미스매치 핵심 단어/구**를 \
추출해 적어라 — 화면(슬라이드/도식/시연 대상)이 가리키는 것과 다르게 발화된 단어 \
하나를 그대로. 화면 텍스트가 아닌 *발화* 텍스트에서 가져와야 ASR transcript와 \
매칭된다. start_sec/end_sec은 대략적이어도 되며, 코드가 ASR transcript에서 이 \
키워드를 찾아 발화 시점으로 좁힌다.
"""

_OTHER_RUBRIC = """당신은 영상 감수 전문가입니다. 아래 영상의 여러 시점 프레임과 음성 \
전사를 검토하고, **컨텍스트 없이 보는 시청자에게 자체적으로 의미가 전달되지 않는 구간**을 \
찾으세요.

판정 기준:
- 무엇을 보여주는지·무엇을 말하는지 파악 가능한가
- 누락된 정보로 인해 의미가 통하지 않는 구간이 있는가

음성 전사가 윈도우 경계("…" 표기)에서 잘려 보이는 것은 끊김이 아니라 인접 구간으로
연속되는 발화이며, 이를 미스매치로 검출하지 마세요.
"""

# vlog는 의도적으로 미등록. 그래프가 vlog에서 이 노드를 호출하지 않으며, 위반 시
# KeyError로 즉시 fail-fast.
_RUBRICS: dict[Category, str] = {
    "lecture": _LECTURE_RUBRIC,
    "other": _OTHER_RUBRIC,
}


@dataclass(frozen=True)
class _FrameSample:
    time_sec: float
    image_b64: str
    transcript_text: str


class _ContentGapIssue(BaseModel):
    start_sec: float = Field(description="문제 구간 시작 시각 (초). 후처리에서 ASR로 좁힘")
    end_sec: float = Field(description="문제 구간 끝 시각 (초). 후처리에서 ASR로 좁힘")
    description: str = Field(
        description="이 구간의 미스매치 내용을 한 문장 한국어로 설명",
        max_length=200,
    )
    mismatch_keyword: str = Field(
        description=(
            "이 구간에서 강사가 발화한 미스매치 핵심 단어/구 — 화면이 가리키는 것과 "
            "다르게 말한 그 단어 하나. 화면 텍스트가 아닌 발화 텍스트 그대로. 코드가 "
            "ASR transcript에서 검색해 발화 시점을 시간 anchor로 잡는다."
        ),
        max_length=50,
    )


# issues max_length로 출력 길이 폭발 차단. 모델이 종료 신호 없이 issue를 무한 생성해
# completion_tokens=16384에 닿아 structured output 파싱이 실패하는 케이스를 막는다.
class _ContentGapResponse(BaseModel):
    issues: list[_ContentGapIssue] = Field(default_factory=list, max_length=5)


def _transcript_around(transcript: list[Word], time_sec: float) -> str:
    """time_sec ± TRANSCRIPT_WINDOW_SEC 범위의 단어들을 텍스트로 결합.

    윈도우 경계 바깥에 word가 있으면 양 끝에 "…" 마커를 붙여 ASR 분할이 만드는 가짜
    누락 신호를 차단한다 — 마커가 없으면 LLM이 윈도우 분할을 "설명 누락"으로 오해해
    hallucinated FP를 발생시킨다.
    """
    lo = time_sec - TRANSCRIPT_WINDOW_SEC
    hi = time_sec + TRANSCRIPT_WINDOW_SEC
    in_window = [w for w in transcript if lo <= w.start <= hi]
    if not in_window:
        return ""
    text = " ".join(w.text for w in in_window).strip()
    if any(w.start < lo for w in transcript):
        text = "… " + text
    if any(w.start > hi for w in transcript):
        text = text + " …"
    return text


def _encode_frame_jpeg(frame) -> str:
    """BGR ndarray → JPEG → base64."""
    h = frame.shape[0]
    if h > MAX_FRAME_HEIGHT:
        scale = MAX_FRAME_HEIGHT / h
        frame = cv2.resize(frame, (int(frame.shape[1] * scale), MAX_FRAME_HEIGHT))
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("프레임 JPEG 인코딩 실패")
    return base64.b64encode(bytes(buffer)).decode("ascii")


def _detect_scene_cuts(video_path: str) -> list[float]:
    """PySceneDetect로 영상 컷 경계 시각 리스트 추출.

    각 씬의 시작 시각을 반환 (첫 씬의 0초는 제외). frame_skip=2로 디코딩 비용 절반.
    검출 실패 시 빈 리스트 (검출은 옵션, 균등 샘플링은 항상 동작).
    """
    try:
        video = open_video(video_path)
        manager = SceneManager()
        manager.add_detector(ContentDetector())
        manager.detect_scenes(video=video, frame_skip=2, show_progress=False)
        scenes = manager.get_scene_list()
    except (OSError, ValueError, RuntimeError):
        return []
    return [s[0].get_seconds() for s in scenes if s[0].get_seconds() > 0.0]


def _uniform_times(duration: float) -> list[float]:
    """SAMPLE_INTERVAL_SEC 간격으로 균등한 시각 리스트 (첫 샘플은 윈도우 중앙)."""
    times: list[float] = []
    t = SAMPLE_INTERVAL_SEC / 2
    while t < duration:
        times.append(t)
        t += SAMPLE_INTERVAL_SEC
    return times


def _merge_sample_times(
    uniform: list[float], cuts: list[float], max_samples: int
) -> list[float]:
    """균등 + 컷 경계 결합. 컷이 정보 우선이라 cuts 먼저 배치 후 균등 중 임계 밖만 추가.

    Cap 적용 시 첫·끝 시각 강제 보존(timeline 양 끝 정보 손실 방지).
    """
    selected: list[float] = sorted(set(cuts))
    for t in sorted(set(uniform)):
        if all(abs(t - s) >= SCENE_DEDUP_THRESHOLD_SEC for s in selected):
            selected.append(t)
    selected.sort()
    if len(selected) > max_samples:
        step = (len(selected) - 1) / (max_samples - 1)
        selected = [selected[round(i * step)] for i in range(max_samples)]
    return selected


def _sample_frames(
    video_path: str, transcript: list[Word]
) -> list[_FrameSample]:
    """균등 시각 + 컷 경계 시각에서 프레임 추출 + transcript 매칭.

    POS_MSEC seek는 GOP 키프레임 경계로 스냅될 수 있어 요청한 t와 실제 디코딩 시각이
    어긋날 수 있음 → cap.get으로 실제 시각을 받아 transcript 윈도우와 정렬.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"미디어 파일 열기 실패: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        sample_times = _merge_sample_times(
            uniform=_uniform_times(duration),
            cuts=_detect_scene_cuts(video_path),
            max_samples=MAX_SAMPLES,
        )
        samples: list[_FrameSample] = []
        for t in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ret, frame = cap.read()
            if not ret:
                break
            actual_t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            samples.append(
                _FrameSample(
                    time_sec=actual_t,
                    image_b64=_encode_frame_jpeg(frame),
                    transcript_text=_transcript_around(transcript, actual_t),
                )
            )
        return samples
    finally:
        cap.release()


def _build_message(samples: list[_FrameSample], rubric: str) -> HumanMessage:
    """프레임 + transcript + rubric을 multi-image HumanMessage로 묶음."""
    content: list[str | dict] = [{"type": "text", "text": rubric}]
    window = int(TRANSCRIPT_WINDOW_SEC)
    for s in samples:
        transcript_block = s.transcript_text or "(해당 구간 발화 없음)"
        content.append(
            {
                "type": "text",
                "text": f"[{s.time_sec:.1f}s ± {window}s 구간 음성 전사]\n{transcript_block}",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{s.image_b64}"},
            }
        )
    content.append(
        {
            "type": "text",
            "text": (
                "위 프레임들에서 발견된 미스매치 구간을 issues 필드에 JSON으로 답하라. "
                "미스매치가 명확한 것만 최대 5건, 각 description은 한 문장. "
                "미스매치가 없으면 빈 리스트."
            ),
        }
    )
    return HumanMessage(content=content)


# LLM이 보고한 [start, end]가 ASR 실제 발화 시점에서 ±이만큼 어긋나도 수용.
# 슬라이드 표시 전체 구간을 잡는 LLM의 공간적 추론 폭을 흡수.
_ASR_ANCHOR_PAD_SEC = 5.0

# 매칭 word의 start/end에 양 옆으로 추가하는 마진. 라벨러가 키워드 발화 전후
# 컨텍스트까지 한 라벨로 묶는 경향을 흡수해 IoU 매칭 폭을 확보.
_ASR_ANCHOR_MARGIN_SEC = 2.0


def _normalize_for_match(text: str) -> str:
    return text.replace(" ", "").strip("\"'`").lower()


def _ngrams(s: str, n: int) -> set[str]:
    return {s[i : i + n] for i in range(len(s) - n + 1)} if len(s) >= n else set()


def _anchor_to_asr(
    issue: _ContentGapIssue, transcript: list[Word]
) -> tuple[float, float] | None:
    """LLM detected 구간 안에서 mismatch_keyword가 발화된 시점으로 좁힘.

    매칭 우선순위:
      (1) 정규화 후 정확 substring 일치
      (2) keyword ≥3자: trigram(3-gram) 일치 — 한국어 ASR이 한두 음절 다르게
          받아쓴 케이스를 흡수하면서 짧은 단어 false positive는 막음
      (3) keyword ≥2자: bigram 일치 — fallback 최후 수단

    매칭 실패 시 None → 호출자가 LLM 원본 [start, end] 그대로 사용.
    매칭 성공 시 양 옆 _ASR_ANCHOR_MARGIN_SEC 패딩 추가.
    """
    keyword = _normalize_for_match(issue.mismatch_keyword)
    if not keyword or not transcript:
        return None

    lo = issue.start_sec - _ASR_ANCHOR_PAD_SEC
    hi = issue.end_sec + _ASR_ANCHOR_PAD_SEC
    # 정규화 결과를 한 번만 계산해 3단계 매칭이 공유.
    in_range = [(w, _normalize_for_match(w.text)) for w in transcript if lo <= w.start <= hi]
    if not in_range:
        return None

    matched = [w for w, norm in in_range if keyword in norm]
    if not matched:
        for n in (3, 2):
            grams = _ngrams(keyword, n)
            if not grams:
                continue
            matched = [w for w, norm in in_range if any(g in norm for g in grams)]
            if matched:
                break
    if not matched:
        return None

    start = max(0.0, min(w.start for w in matched) - _ASR_ANCHOR_MARGIN_SEC)
    end = max(w.end for w in matched) + _ASR_ANCHOR_MARGIN_SEC
    return start, end


async def detect_content_gap_events(
    video_path: str,
    transcript: list[Word],
    category: Category,
) -> list[ContentGapEvent]:
    """영상 + transcript + 카테고리 → 내용 공백 이벤트 리스트.

    그래프가 lecture·other 카테고리에서만 이 노드를 호출한다. category는 rubric 선택용.
    LLM 응답의 [start, end]는 mismatch_keyword가 ASR transcript에서 발화된 시점으로
    좁혀 반환한다 — 의미는 LLM, 시간 anchor는 ASR이 책임.
    """
    samples = _sample_frames(video_path, transcript)
    if not samples:
        return []
    message = _build_message(samples, _RUBRICS[category])

    # max_tokens=1024: issues 5건 × ~200자로 충분. structured output length limit
    # 도달로 인한 파싱 실패 차단.
    model = get_chat_model(model="gpt-4o", temperature=0.0, max_tokens=1024)
    structured = model.with_structured_output(_ContentGapResponse)
    response = cast(_ContentGapResponse, await structured.ainvoke([message]))

    events: list[ContentGapEvent] = []
    for issue in response.issues:
        anchored = _anchor_to_asr(issue, transcript)
        if anchored is None:
            start, end = issue.start_sec, issue.end_sec
        else:
            start, end = anchored
        events.append(
            ContentGapEvent(start=start, end=end, description=issue.description)
        )
    return events
