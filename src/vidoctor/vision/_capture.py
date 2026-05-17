"""vision 모듈 공용 cv2.VideoCapture 라이프사이클 + 프레임 인코딩.

- `open_capture`: `cv2.VideoCapture` 컨텍스트 매니저 — open 실패 시 명확한 에러 + 항상 release.
- `encode_frame_jpeg`: BGR ndarray → 다운스케일 → JPEG → base64. max_height/quality만
  호출부별로 다르고 알고리즘 동일.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator
from contextlib import contextmanager

import cv2


@contextmanager
def open_capture(video_path: str) -> Iterator[cv2.VideoCapture]:
    """cv2.VideoCapture 열고 finally에서 release. 실패 시 한국어 RuntimeError."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"미디어 파일 열기 실패: {video_path}")
    try:
        yield cap
    finally:
        cap.release()


def encode_frame_jpeg(
    frame: cv2.typing.MatLike, *, max_height: int, quality: int
) -> str:
    """BGR ndarray → 비율 유지 다운스케일(height>max_height일 때) → JPEG → base64."""
    h = frame.shape[0]
    if h > max_height:
        scale = max_height / h
        frame = cv2.resize(frame, (int(frame.shape[1] * scale), max_height))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("프레임 JPEG 인코딩 실패")
    return base64.b64encode(bytes(buf)).decode("ascii")
