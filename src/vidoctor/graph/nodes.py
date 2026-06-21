"""5차원 분석 노드.

각 노드는 무거운 라이브러리(WhisperX, MediaPipe 등)를 함수 안에서 lazy import —
비활성 차원의 모듈 로딩 비용 회피. 카테고리별 활성/비활성은 pipeline.py가 결정.

- transcribe: WhisperX (faster-whisper-large-v3-turbo + wav2vec2 forced alignment)
- detect_filler: 한국어 filler 사전 + 정규식
- detect_cps: Net CPS 슬라이딩 윈도우 (5s/1s, pause >200ms 제외)
- detect_dead_zone: Silero VAD + Optical flow magnitude per-frame max
- detect_gaze: MediaPipe FaceLandmarker + cv2.solvePnP head pose
- detect_content_gap: GPT-4o Vision multi-image + ASR
- generate_suggestions: GPT-4o-mini로 5차원 finding 통합 (suggestions.py)
"""

import asyncio
import logging

from vidoctor.graph.state import AnalysisState

_log = logging.getLogger(__name__)

# 오디오 트랙 없는 영상은 whisperx의 ffmpeg가 이 문구가 든 RuntimeError를 던진다.
_NO_AUDIO_MARKER = "does not contain any stream"
# 무음·오디오 트랙 없음뿐 아니라 ASR 인식 실패도 포함하므로 '없다'고 단정하지 않는다.
_NO_TRANSCRIPT_MSG = (
    "영상에서 음성을 인식하지 못해 분석할 수 없습니다. 말소리가 또렷한 영상인지 확인해 주세요."
)
_FOREIGN_MSG = "한국어 음성만 분석할 수 있습니다. 한국어 영상을 올려주세요."
# 비한국어로 '확신할' 때만 거부 — 음악·모호한 입력은 통과시켜 무음 게이트(빈 transcript)가
# 처리하게 한다(외국어 오분류 방지). 한국어는 tiny가 확신 있게 잡아 오거부 위험이 낮다.
_LANG_MIN_CONFIDENCE = 0.5


async def transcribe(state: AnalysisState) -> dict:
    from vidoctor.audio.transcribe import LANGUAGE, detect_language, transcribe_video
    from vidoctor.errors import SafeError

    try:
        words, audio = await transcribe_video(state["video_path"])
        # 전사가 디코딩한 audio를 재사용해 언어 감지 — 비한국어면 결과를 버리고 중단.
        lang, lang_prob = await detect_language(audio)
        if lang != LANGUAGE and lang_prob >= _LANG_MIN_CONFIDENCE:
            raise SafeError(_FOREIGN_MSG)
    except RuntimeError as e:
        if _NO_AUDIO_MARKER in str(e):
            raise SafeError(_NO_TRANSCRIPT_MSG) from e
        raise
    # 음성을 인식 못하면 음성 기반 차원(filler·cps·content_gap·dead_zone)이 무의미해진다.
    # 빈 결과로 완료하면 '발견 0건'이 '완벽한 영상'으로 오인되므로 명시적으로 중단한다.
    if not words:
        raise SafeError(_NO_TRANSCRIPT_MSG)
    return {"transcript": words, "audio_16k": audio}


async def detect_filler(state: AnalysisState) -> dict:
    from vidoctor.audio.filler import detect_filler_events

    transcript = state.get("transcript", [])
    return {"fillers": detect_filler_events(transcript)}


async def detect_cps(state: AnalysisState) -> dict:
    """카테고리별 분기: vlog는 F0 multi-feature 결합, 그 외는 cps 단독.

    vlog는 배경 노이즈로 ASR이 오염되므로 F0(메인 화자 voiced 톤) 결합으로 노이즈 cut
    — F1 0.533 → 0.667. lecture는 노이즈 적고 톤 단조로워 F0 결합 시 오히려 라벨 cut.
    F0 추출은 transcribe가 디코딩한 audio_16k 재사용 — librosa.load의 mp4 fallback
    (audioread+ffmpeg) 비용 회피.
    """
    from vidoctor.audio.cps import detect_cps_anomalies, detect_cps_with_audio

    transcript = state.get("transcript", [])
    if state["category"] != "vlog":
        return {"cps_anomalies": detect_cps_anomalies(transcript)}

    audio = state.get("audio_16k")
    if audio is None:
        # transcribe가 audio_16k를 채우지 못한 invariant 위반 — F0 없는 단독 detector로 폴백.
        return {"cps_anomalies": detect_cps_anomalies(transcript)}
    events = await asyncio.to_thread(detect_cps_with_audio, transcript, audio)
    return {"cps_anomalies": events}


async def detect_dead_zone(state: AnalysisState) -> dict:
    from vidoctor.vision.dead_zone import detect_dead_zone_events

    events = await detect_dead_zone_events(
        state["video_path"], state["category"], audio=state.get("audio_16k")
    )
    return {"dead_zones": events}


async def detect_gaze(state: AnalysisState) -> dict:
    from vidoctor.vision.gaze import detect_gaze_events

    events = await detect_gaze_events(state["video_path"])
    return {"gaze_issues": events}


async def detect_content_gap(state: AnalysisState) -> dict:
    from vidoctor.vision.content_gap import detect_content_gap_events

    transcript = state.get("transcript", [])
    events, metrics = await detect_content_gap_events(
        state["video_path"], transcript, state["category"]
    )
    return {"content_gaps": events, "step_metrics": [metrics]}


async def generate_suggestions(state: AnalysisState) -> dict:
    from vidoctor.suggestions import build_suggestions

    # 최종 노드라 여기서 예외가 나면 앞서 검출된 finding까지 통째로 유실된다 — 격리.
    try:
        suggestions, metrics = await build_suggestions(state)
        return {"suggestions": suggestions, "step_metrics": [metrics]}
    except Exception:
        _log.exception("개선 제안 생성 실패 — 검출 결과는 보존")
        return {"suggestions": []}
