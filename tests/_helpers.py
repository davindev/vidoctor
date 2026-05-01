"""테스트 공통 헬퍼."""

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from vidoctor.graph.state import Word


def _w(text: str, start: float, end: float, score: float | None = 0.9) -> Word:
    """단위 테스트용 Word 인스턴스 팩토리."""
    return Word(text=text, start=start, end=end, score=score)


def write_video(
    path: Path,
    duration_sec: float,
    frame_fn: Callable[[int, float], np.ndarray],
    *,
    fps: int = 24,
    size: tuple[int, int] = (320, 240),
) -> None:
    """frame_fn(idx, t)로 합성 영상 mp4 생성. dead_zone·content_gap 테스트 공용."""
    width, height = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(int(duration_sec * fps)):
        t = i / fps
        writer.write(frame_fn(i, t))
    writer.release()
