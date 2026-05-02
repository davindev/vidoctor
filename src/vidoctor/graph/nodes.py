"""5차원 분석 노드.

- transcribe: WhisperX (faster-whisper-large-v3-turbo + wav2vec2 forced alignment) ← 구현됨
- detect_filler: 한국어 filler 사전 + 정규식
- detect_cps: Net CPS 슬라이딩 윈도우 (5s/1s, pause >200ms 제외)
- detect_dead_zone: OpenCV diff + SSIM + ASR 무발화
- detect_gaze: MediaPipe Tasks FaceLandmarker + cv2.solvePnP head pose (강의만)
- detect_content_gap: GPT-4o Vision multi-image batch + ASR (강의·기타만)
- generate_suggestions: GPT-4o-mini로 finding 통합 → 개선 제안
"""

import asyncio

from vidoctor.graph.state import AnalysisState


async def transcribe(state: AnalysisState) -> dict:
    from vidoctor.audio.transcribe import transcribe_video

    words = await transcribe_video(state["video_path"])
    return {"transcript": words}


async def detect_filler(state: AnalysisState) -> dict:
    from vidoctor.audio.filler import detect_filler_events

    transcript = state.get("transcript", [])
    return {"fillers": detect_filler_events(transcript)}


async def detect_cps(state: AnalysisState) -> dict:
    from vidoctor.audio.cps import detect_cps_anomalies

    transcript = state.get("transcript", [])
    return {"cps_anomalies": detect_cps_anomalies(transcript)}


async def detect_dead_zone(state: AnalysisState) -> dict:
    from vidoctor.vision.dead_zone import detect_dead_zone_events

    transcript = state.get("transcript", [])
    events = await detect_dead_zone_events(
        state["video_path"], transcript, state["category"]
    )
    return {"dead_zones": events}


async def detect_gaze(state: AnalysisState) -> dict:
    from vidoctor.vision.gaze import detect_gaze_events

    events = await detect_gaze_events(state["video_path"], state["category"])
    return {"gaze_issues": events}


async def detect_content_gap(state: AnalysisState) -> dict:
    from vidoctor.vision.content_gap import detect_content_gap_events

    transcript = state.get("transcript", [])
    events = await detect_content_gap_events(
        state["video_path"], transcript, state["category"]
    )
    return {"content_gaps": events}


async def generate_suggestions(state: AnalysisState) -> dict:
    await asyncio.sleep(0.01)
    return {"suggestions": []}
