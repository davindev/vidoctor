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

from vidoctor.graph.state import AnalysisState


async def transcribe(state: AnalysisState) -> dict:
    from vidoctor.audio.transcribe import transcribe_video

    words, audio = await transcribe_video(state["video_path"])
    return {"transcript": words, "audio_16k": audio}


async def detect_filler(state: AnalysisState) -> dict:
    from vidoctor.audio.filler import detect_filler_events

    transcript = state.get("transcript", [])
    return {"fillers": detect_filler_events(transcript)}


async def detect_cps(state: AnalysisState) -> dict:
    """카테고리별 분기: vlog는 F0 multi-feature 결합, 그 외는 cps 단독.

    vlog는 배경 노이즈로 ASR이 오염되므로 F0(메인 화자 voiced 톤) 결합으로 노이즈 cut
    — F1 0.533 → 0.667. lecture는 노이즈 적고 톤 단조로워 F0 결합 시 오히려 라벨 cut.
    """
    from vidoctor.audio.cps import detect_cps_anomalies, detect_cps_with_audio

    transcript = state.get("transcript", [])
    if state["category"] != "vlog":
        return {"cps_anomalies": detect_cps_anomalies(transcript)}

    events = await asyncio.to_thread(
        detect_cps_with_audio, transcript, state["video_path"]
    )
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

    suggestions, metrics = await build_suggestions(state)
    return {"suggestions": suggestions, "step_metrics": [metrics]}
