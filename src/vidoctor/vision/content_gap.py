"""내용 공백 검출 — GPT-4o Vision multi-image batch + 카테고리 rubric.

브이로그는 비활성 (일상 기록·감정 공유 영상에 "설명 충분성" 평가 부적합).
강의는 "주제 추정 가능?" rubric, 기타는 "자체 의미 전달?" rubric.

알고리즘:
1. 영상에서 SAMPLE_INTERVAL_SEC(30s) 균등 + PySceneDetect 컷 경계 결합 프레임 시각 추출
2. dedup (인접 시각 5s 이내는 하나만) + MAX_SAMPLES 캡으로 비용 통제
3. 각 프레임을 720p 다운스케일 + JPEG 인코딩 + base64
4. 각 프레임 시각 ± WINDOW_SEC(15s) transcript 추출
5. 프레임·transcript·rubric을 한 multi-image 메시지로 묶어 GPT-4o Vision 호출
6. Pydantic structured output(_ContentGapResponse)으로 파싱 → ContentGapEvent

상수는 휴리스틱 시작값. 골든셋 라벨링 후 우선 튜닝:
  1순위: SAMPLE_INTERVAL_SEC / MAX_SAMPLES — 영상 길이·비용 대비 recall
  2순위: 카테고리별 rubric 본문 — Cohen's κ ≥ 0.6 달성까지 프롬프트 엔지니어링
  3순위: SCENE_DEDUP_THRESHOLD_SEC — 컷 ↔ 균등 샘플 충돌 처리
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
해당 구간의 음성 전사를 검토하고, **시청자가 주제를 따라가기 어렵거나 정보가 부족한 \
구간**을 찾아내세요.

판정 기준:
- 슬라이드/화면만 봐도 무엇에 관한 강의인지 추정 가능한가
- 음성과 화면 내용이 일관성 있게 연결되는가
- 갑작스러운 비약이나 누락된 설명은 없는가
- 같은 내용의 무의미한 반복은 없는가
"""

_OTHER_RUBRIC = """당신은 영상 감수 전문가입니다. 아래 영상의 여러 시점 프레임과 음성 \
전사를 검토하고, **컨텍스트 없이 보는 시청자에게 자체적으로 의미가 전달되지 않는 구간**을 \
찾으세요.

판정 기준:
- 무엇을 보여주는지·무엇을 말하는지 파악 가능한가
- 누락된 정보로 인해 의미가 통하지 않는 구간이 있는가
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
    """LLM 응답 단일 항목."""

    start_sec: float = Field(description="문제 구간 시작 시각 (초)")
    end_sec: float = Field(description="문제 구간 끝 시각 (초)")
    description: str = Field(description="이 구간의 문제점을 한 문장 한국어로 설명")


class _ContentGapResponse(BaseModel):
    """LLM이 반환할 전체 응답 스키마."""

    issues: list[_ContentGapIssue] = Field(default_factory=list)


def _transcript_around(transcript: list[Word], time_sec: float) -> str:
    """time_sec ± TRANSCRIPT_WINDOW_SEC 범위의 단어들을 텍스트로 결합."""
    lo = time_sec - TRANSCRIPT_WINDOW_SEC
    hi = time_sec + TRANSCRIPT_WINDOW_SEC
    in_window = [w.text for w in transcript if lo <= w.start <= hi]
    return " ".join(in_window).strip()


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

    각 씬의 시작 시각을 반환 (첫 씬의 0초는 제외).
    frame_skip=2로 디코딩 비용 절반 (3분 영상 ~3s → ~1.5s).
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
        # i=0 → 첫 원소, i=max_samples-1 → 마지막 원소 보장
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
            "text": "위 프레임들에서 발견된 문제 구간을 issues 필드에 JSON으로 답하라. "
            "문제가 없으면 빈 리스트.",
        }
    )
    return HumanMessage(content=content)


async def detect_content_gap_events(
    video_path: str,
    transcript: list[Word],
    category: Category,
) -> list[ContentGapEvent]:
    """영상 + transcript + 카테고리 → 내용 공백 이벤트 리스트.

    그래프가 lecture·other 카테고리에서만 이 노드를 호출한다. category는 rubric 선택용.
    """
    samples = _sample_frames(video_path, transcript)
    if not samples:
        return []

    message = _build_message(samples, _RUBRICS[category])

    model = get_chat_model(model="gpt-4o", temperature=0.0)
    structured = model.with_structured_output(_ContentGapResponse)
    response = cast(_ContentGapResponse, await structured.ainvoke([message]))

    return [
        ContentGapEvent(
            start=issue.start_sec,
            end=issue.end_sec,
            description=issue.description,
        )
        for issue in response.issues
    ]
