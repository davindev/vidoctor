"""F0(피치) 신호 추출 — cps 차원 multi-feature 보조 신호.

라벨러가 인지하는 "속사포"는 발화 속도뿐 아니라 *톤 상승*을 동반한다는 골든셋 분석
결과를 반영. cps z-score만으론 정상 발화의 단순 빠름과 라벨러가 인지하는 속사포를
분리하지 못해, F0 평균·범위를 추가 신호로 사용.

추출은 librosa pYIN (probabilistic YIN, voiced/unvoiced 자동 판정). 한국어 화자
F0 범위 80~400Hz로 한정해 noise floor 컷.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

# 한국어 화자 평균 F0 범위. 남성 80~180Hz, 여성 180~400Hz를 모두 포함하도록 80~400Hz.
# 이 범위 밖은 noise·자음 기반 false positive로 컷.
F0_MIN_HZ = 80.0
F0_MAX_HZ = 400.0

# pYIN 분석 파라미터. 16kHz 다운샘플링 후 frame 2048 / hop 512 → 32fps F0 시계열.
# 이 정밀도면 5초 윈도우당 ~160개 voiced 프레임 표본 → 평균·range 안정.
F0_FRAME_LENGTH = 2048
F0_HOP_LENGTH = 512
F0_SAMPLE_RATE = 16000

# 윈도우 안 voiced 프레임 최소 표본. 너무 적으면 평균·range가 noise.
MIN_VOICED_FRAMES = 5


@dataclass(frozen=True)
class WindowPitch:
    """한 윈도우 시간 영역의 F0 통계."""

    start: float
    end: float
    f0_mean: float
    f0_std: float
    f0_range: float


def extract_pitch_track(audio_path: str) -> tuple[np.ndarray, np.ndarray]:
    """오디오 파일에서 pYIN으로 F0 시계열 추출. (f0, times) 반환.

    f0는 unvoiced 프레임에서 NaN. 호출자가 voiced만 골라 윈도우 통계 계산.
    """
    y, _ = librosa.load(audio_path, sr=F0_SAMPLE_RATE, mono=True)
    f0, _, _ = librosa.pyin(
        y,
        fmin=F0_MIN_HZ,
        fmax=F0_MAX_HZ,
        sr=F0_SAMPLE_RATE,
        frame_length=F0_FRAME_LENGTH,
        hop_length=F0_HOP_LENGTH,
    )
    times = librosa.times_like(f0, sr=F0_SAMPLE_RATE, hop_length=F0_HOP_LENGTH)
    return f0, times


def window_pitch_features(
    f0: np.ndarray,
    times: np.ndarray,
    windows: list[tuple[float, float]],
) -> list[WindowPitch | None]:
    """각 윈도우 (start, end)에 대해 voiced F0 평균·std·range 계산.

    voiced 프레임이 MIN_VOICED_FRAMES 미만이면 None — multi-feature detector가 그 윈도우는
    F0 신호 없는 것으로 처리해야 한다.
    """
    features: list[WindowPitch | None] = []
    for start, end in windows:
        mask = (times >= start) & (times < end)
        voiced = f0[mask]
        voiced = voiced[~np.isnan(voiced)]
        if len(voiced) < MIN_VOICED_FRAMES:
            features.append(None)
            continue
        features.append(
            WindowPitch(
                start=start,
                end=end,
                f0_mean=float(np.mean(voiced)),
                f0_std=float(np.std(voiced)),
                f0_range=float(np.max(voiced) - np.min(voiced)),
            )
        )
    return features
