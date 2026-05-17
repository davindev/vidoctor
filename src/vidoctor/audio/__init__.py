"""음성 도메인 — WhisperX ASR + filler/CPS 검출 + F0(pitch) 추출."""

from vidoctor.audio.transcribe import transcribe_video

__all__ = ["transcribe_video"]
