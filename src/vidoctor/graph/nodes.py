"""5차원 분석 노드 placeholder.

각 모듈을 실제 구현할 때 이 파일의 함수를 채운다:
- transcribe: WhisperX (faster-whisper-large-v3-turbo + wav2vec2 forced alignment)
- detect_filler: 한국어 filler 사전 + 정규식
- detect_cps: Net CPS 슬라이딩 윈도우 (5s/1s, pause >200ms 제외)
- detect_dead_zone: OpenCV diff + SSIM + ASR 무발화
- detect_gaze: MediaPipe Face Mesh iris + cv2.solvePnP head pose (강의만)
- detect_content_gap: GPT-4o Vision multi-image batch + ASR (강의·기타만)
- generate_suggestions: GPT-4o-mini로 finding 통합 → 개선 제안
"""

import asyncio

from vidoctor.graph.state import AnalysisState


async def transcribe(state: AnalysisState) -> dict:
    await asyncio.sleep(0.01)
    return {"transcript": []}


async def detect_filler(state: AnalysisState) -> dict:
    await asyncio.sleep(0.01)
    return {"fillers": []}


async def detect_cps(state: AnalysisState) -> dict:
    await asyncio.sleep(0.01)
    return {"cps_anomalies": []}


async def detect_dead_zone(state: AnalysisState) -> dict:
    await asyncio.sleep(0.01)
    return {"dead_zones": []}


async def detect_gaze(state: AnalysisState) -> dict:
    if state.get("category") != "lecture":
        return {"gaze_issues": []}
    await asyncio.sleep(0.01)
    return {"gaze_issues": []}


async def detect_content_gap(state: AnalysisState) -> dict:
    if state.get("category") == "vlog":
        return {"content_gaps": []}
    await asyncio.sleep(0.01)
    return {"content_gaps": []}


async def generate_suggestions(state: AnalysisState) -> dict:
    await asyncio.sleep(0.01)
    return {"suggestions": []}
