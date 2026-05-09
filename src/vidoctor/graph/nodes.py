"""5차원 분석 노드.

카테고리별 차원 활성/비활성은 `pipeline.py`의 conditional edge가 결정하며, 이 모듈의
detection 노드는 자기 차원만 책임진다 — detect_gaze / detect_content_gap이
비활성 카테고리에서 호출되지 않는 것은 그래프가 보장.

- transcribe: WhisperX (faster-whisper-large-v3-turbo + wav2vec2 forced alignment)
- detect_filler: 한국어 filler 사전 + 정규식
- detect_cps: Net CPS 슬라이딩 윈도우 (5s/1s, pause >200ms 제외)
- detect_dead_zone: Silero VAD 무발화 + Optical flow magnitude per-frame max
- detect_gaze: MediaPipe Tasks FaceLandmarker + cv2.solvePnP head pose
- detect_content_gap: GPT-4o Vision multi-image batch + ASR
- generate_suggestions: GPT-4o-mini로 5차원 finding 통합 → 개선 제안 (suggestions.py)
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
    """카테고리별 정책 분기: vlog는 F0 multi-feature 결합, lecture/other는 cps 단독.

    vlog는 야외·일상 녹화 환경에서 배경 노이즈·다른 화자 발화가 ASR 토큰에 섞여 들어
    cps 측정을 오염한다. F0 결합은 메인 화자만이 강한 voiced 톤 신호를 갖는 특성을 이용해
    노이즈 영역을 자동 cut — vlog F1 0.533 → 0.667(P 0.500 → 0.800). lecture는 통제된
    녹화 환경이라 노이즈 거의 없고 발화 톤이 단조로워 F0 결합 시 라벨이 F0 임계 미달로
    오히려 cut → cps 단독이 robust. other는 도메인 다양성으로 보수 fallback.
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
        state["video_path"], state["category"]
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
