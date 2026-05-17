"""강의 영상 시선 이탈 검출 — MediaPipe FaceLandmarker + cv2.solvePnP head pose.

영상 전체 yaw/pitch median을 정면 baseline으로 차감 후 deviation 임계 적용.
무캘리 cv2.solvePnP + decomposeProjectionMatrix는 카메라·focal·euler 분해 컨벤션에 따라
영상별 systematic offset(±수~십도)을 만들어 절대 임계로는 영상마다 분리 성능이 흔들림 —
강사가 대부분 정면 응시한다는 강의 도메인 가정 위에서 median이 robust baseline.

FaceLandmarker.task 모델은 첫 호출 시 자동 다운로드(`models/`). graph가 lecture에서만 호출.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from vidoctor.config import ROOT
from vidoctor.graph.state import Direction, GazeEvent
from vidoctor.vision._capture import open_capture

if TYPE_CHECKING:
    from numpy.typing import NDArray

_MODEL_DIR = ROOT / "models"

_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
_LANDMARKER_MODEL_PATH = _MODEL_DIR / "face_landmarker.task"
# Google MediaPipe CDN의 `latest` 경로는 무공지 교체될 수 있어 silent drift 위험. hash를
# 코드에 박아 받은 파일을 검증하고, 불일치 시 fail-loud + 골든셋 재평가 사이클로 의식적 갱신.
_LANDMARKER_MODEL_SHA256 = (
    "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
)

# BlazeFace short-range: 가까운 거리(~2m)의 얼굴 검출용. FaceLandmarker는 큰 얼굴 가정으로
# 학습돼 강의 영상의 작은 웹캠 영역(전체의 5% 이하) 얼굴을 못 잡는 경우가 많아, 별도 단계로
# BlazeFace로 ROI를 먼저 추정한다.
_DETECTOR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
_DETECTOR_MODEL_PATH = _MODEL_DIR / "blaze_face_short_range.tflite"
_DETECTOR_MODEL_SHA256 = (
    "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f"
)

# 일반 성인 얼굴 6점 3D 모델 (mm). solvePnP 표준 reference.
# (코끝, 턱끝, 왼눈 좌측 corner, 오른눈 우측 corner, 입 좌측, 입 우측)
_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float64,
)

# MediaPipe Face Mesh 표준 478 landmark에서 위 6점에 대응하는 인덱스
# (canonical face mesh는 legacy/Tasks 동일).
_LANDMARK_IDS: tuple[int, ...] = (1, 152, 33, 263, 61, 291)

# baseline 차감 후 deviation에 적용하는 임계 (deg). lecture 골든셋 측정에서 정상 응시
# deviation의 abs_p95가 yaw 4.3° / pitch 6.1°로, 임계 10/12°가 일반 변동과 충분히 분리.
# 양 차원 OR 조건이라 한쪽이라도 임계 넘으면 이탈로 본다.
YAW_THRESHOLD_DEG = 10.0
PITCH_THRESHOLD_DEG = 12.0

# 프레임 샘플링 fps. 30fps 전수 처리는 오버킬, 5fps면 200ms 해상도로 충분.
SAMPLE_FPS = 5.0

# 최소 지속 시간(s). 자연스런 짧은 head turn·깜빡임을 노이즈로 제거하되, 짧게 휙 이동하는
# 시선 이탈도 잡히도록 0.6s. MERGE_GAP과 함께 라벨 수 초 구간 안 여러 짧은 이탈을 한
# 이벤트로 묶어 IoU 매칭 손실 회피.
MIN_DURATION_SEC = 0.6

# 인접 이상 프레임 사이 정면 복귀를 같은 이벤트로 묶는 갭. 라벨러는 보통 수 초 단위
# 구간으로 묶고 detector는 frame 단위라 단위 mismatch가 IoU를 깎는다. 1.0s까지 묶으면
# 4초 라벨 안 짧은 이탈 여러 개가 한 이벤트로 합쳐져 매칭 안정.
MERGE_GAP_SEC = 1.0


@dataclass(frozen=True)
class GazeConfig:
    """gaze 검출 임계 묶음. eval sweep이 모듈 globals를 mutate하던 hack을 대체 — 임계는
    이 객체로 전달해 samples_to_events / _is_off가 순수 함수가 된다."""

    yaw_threshold_deg: float = YAW_THRESHOLD_DEG
    pitch_threshold_deg: float = PITCH_THRESHOLD_DEG
    min_duration_sec: float = MIN_DURATION_SEC
    merge_gap_sec: float = MERGE_GAP_SEC


_DEFAULT_CONFIG = GazeConfig()

# 자동 ROI 추정 파라미터. 강의 화자 위치는 거의 고정이라 첫 5초 내 BlazeFace 검출 결과만으로
# 안정 ROI 결정 가능. 머리 회전 시 얼굴이 ROI 밖으로 나가지 않도록 detected bbox의 1.6배로 확장.
ROI_DETECTION_WINDOW_SEC = 5.0
ROI_MARGIN_FACTOR = 1.6

# 4코너 폴백 search region 크기 (영상 크기 대비 비율). 작은 웹캠도 입력 대비 비율이 충분히
# 커지도록 0.4 채택 (전체 ~16% 면적). 너무 작으면 머리 끝이 region 밖으로 나가 false negative.
_CORNER_REGION_RATIO = 0.4

# (yaw_sign, pitch_sign) → Direction. 임계 외/내를 -1/0/+1로 양자화한 뒤 lookup.
# 방향 문자열 join을 직접 쓰면 Literal 좁히기가 안 돼 lookup table 채택.
_DIRECTIONS: dict[tuple[int, int], Direction] = {
    (0, 0): "front",
    (1, 0): "right",
    (-1, 0): "left",
    (0, 1): "down",
    (0, -1): "up",
    (1, 1): "right_down",
    (1, -1): "right_up",
    (-1, 1): "left_down",
    (-1, -1): "left_up",
}


@dataclass(frozen=True)
class _VideoMeta:
    """VideoCapture에서 한 번에 추출하는 영상 메타. ROI 검출·메인 샘플링이 공유."""

    fps: float
    width: int
    height: int
    total_frames: int

    @property
    def sample_step(self) -> int:
        return max(int(round(self.fps / SAMPLE_FPS)), 1)


@dataclass(frozen=True)
class PoseSample:
    """프레임 단위 head pose 측정값. is_off/direction은 임계 의존이라 raw를 보존."""

    t: float
    yaw: float
    pitch: float


@dataclass(frozen=True)
class _ROI:
    """웹캠 영역 사각형 (원본 영상 좌표계, 픽셀 단위)."""

    x: int
    y: int
    w: int
    h: int


def _read_video_meta(cap: cv2.VideoCapture) -> _VideoMeta:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    return _VideoMeta(
        fps=fps,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )


def _solve_head_pose(
    points_2d: NDArray[np.float64], width: int, height: int
) -> tuple[float, float] | None:
    """6개 face landmark → (yaw, pitch) 도. 실패 시 None.

    카메라 내부 파라미터는 영상 width를 focal length로 가정하는 무캘리 근사.
    뷰포트 차이로 절대값엔 ±2~5° 오차 있을 수 있으나 임계 ±15~20°는 충분히 분리됨.
    """
    focal = float(width)
    center = (width / 2.0, height / 2.0)
    camera_matrix = np.array(
        [
            [focal, 0, center[0]],
            [0, focal, center[1]],
            [0, 0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        _MODEL_POINTS,
        points_2d,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    projection = np.hstack((rmat, tvec))
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(projection)
    pitch_raw, yaw_raw = float(euler.flatten()[0]), float(euler.flatten()[1])
    return _normalize_pose_angle(yaw_raw), _normalize_pose_angle(pitch_raw)


def _normalize_pose_angle(deg: float) -> float:
    """카메라 정면 기준 [-90, 90]도로 정규화.

    decomposeProjectionMatrix는 회전 분해 컨벤션상 yaw/pitch를 ±180° 부근으로 토하기도
    한다. solvePnP 입력이 정면 응시일 때 우리 6점 model에선 결과가 ±170~180° 부근.
    이 함수는 표준 [-180, 180] wrap이 아니라, "정면 = 0°" 기준으로 부호를 보존한
    [-90, 90] reflection이다. 카메라 좌표계 규약상 ±90° 바깥은 의미가 없어 mirror로
    접는 것이 옳다.
    """
    while deg > 180.0:
        deg -= 360.0
    while deg < -180.0:
        deg += 360.0
    if deg > 90.0:
        return 180.0 - deg
    if deg < -90.0:
        return -180.0 - deg
    return deg


def _label_direction(yaw: float, pitch: float, cfg: GazeConfig) -> Direction:
    h = 1 if yaw > cfg.yaw_threshold_deg else -1 if yaw < -cfg.yaw_threshold_deg else 0
    v = 1 if pitch > cfg.pitch_threshold_deg else -1 if pitch < -cfg.pitch_threshold_deg else 0
    return _DIRECTIONS[(h, v)]


def _is_off(sample: PoseSample, cfg: GazeConfig) -> bool:
    return abs(sample.yaw) > cfg.yaw_threshold_deg or abs(sample.pitch) > cfg.pitch_threshold_deg


def samples_to_events(
    samples: list[PoseSample], cfg: GazeConfig = _DEFAULT_CONFIG
) -> list[GazeEvent]:
    """is_off 연속 구간을 GazeEvent로 묶음. 짧은 정면 복귀는 cfg.merge_gap_sec까지 같은 이벤트."""
    events: list[GazeEvent] = []
    cur_start: float | None = None
    cur_end = 0.0
    cur_dir: Direction = "front"
    last_off_t = -math.inf

    def flush() -> None:
        nonlocal cur_start
        if cur_start is None:
            return
        duration = cur_end - cur_start
        if duration >= cfg.min_duration_sec:
            events.append(GazeEvent(start=cur_start, end=cur_end, direction=cur_dir))
        cur_start = None

    for sample in samples:
        if _is_off(sample, cfg):
            if cur_start is None:
                cur_start = sample.t
                cur_dir = _label_direction(sample.yaw, sample.pitch, cfg)
            cur_end = sample.t
            last_off_t = sample.t
        elif cur_start is not None and (sample.t - last_off_t) > cfg.merge_gap_sec:
            flush()

    flush()
    return events


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=2)
def _ensure_model(url: str, path: Path, expected_sha256: str) -> Path:
    """모델 파일을 캐시 + SHA256으로 무결성 검증.

    실패 정책:
    - 다운로드 자체 실패: urllib 예외 전파 (호출자가 catch).
    - 받은 파일 hash 불일치: tmp 삭제, 최종 경로 보존, RuntimeError. Google이 `latest`
      교체했을 가능성 → 의도적 업그레이드면 골든셋 재평가 후 상수 갱신.
    - 기존 파일 hash 불일치: 자동 재다운로드 안 함(의도적 교체 silent 덮어쓰기 회피).
      RuntimeError 메시지에 `rm <path>` 후 재실행 안내.

    원자성: tmp에 받아 검증 통과 후 path.replace로 atomic rename. 중간 실패에도 최종
    경로는 늘 valid 상태(옛 정상본 또는 새 정상본).

    lru_cache: `sample_video_pose`가 영상마다 호출되며 _ensure_model을 부르는데 같은
    인자(2종)면 결과 동일하므로 프로세스 수명 동안 검증 1회면 충분.
    """
    if path.exists():
        actual = _sha256_of(path)
        if actual == expected_sha256:
            return path
        raise RuntimeError(
            f"기존 모델 hash 불일치 ({path.name}): "
            f"expected={expected_sha256[:12]}…, actual={actual[:12]}…. "
            f"`rm {path}` 후 재실행하면 자동 재다운로드."
        )

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    urllib.request.urlretrieve(url, tmp)

    actual = _sha256_of(tmp)
    if actual != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"받은 모델 hash 불일치 ({path.name}): {url}이 교체됐을 수 있음. "
            f"expected={expected_sha256}, actual={actual}. "
            f"의도적 업그레이드면 골든셋 재평가 후 코드 상수 갱신."
        )
    tmp.replace(path)
    return path


@lru_cache(maxsize=1)
def _get_face_detector():  # noqa: ANN202
    """BlazeFace 인스턴스를 프로세스 수명 동안 1회 생성·캐시.

    IMAGE 모드라 호출 간 timestamp 의존 없어 캐시 안전.
    (FaceLandmarker는 VIDEO 모드 timestamp 단조 증가 요건이 있어 호출별 새 인스턴스 유지.)
    """
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    options = mp_vision.FaceDetectorOptions(
        base_options=mp_tasks.BaseOptions(
            model_asset_path=str(
                _ensure_model(
                    _DETECTOR_MODEL_URL, _DETECTOR_MODEL_PATH, _DETECTOR_MODEL_SHA256
                )
            )
        ),
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    return mp_vision.FaceDetector.create_from_options(options)


def _search_regions(meta: _VideoMeta) -> list[tuple[int, int, int, int]]:
    """전체 프레임 → 4코너 순 search region. 강의 영상은 보통 슬라이드 + 한 코너 웹캠 형식이라
    전체 프레임에서 못 잡으면 코너에서 잡힘.
    """
    cw = int(meta.width * _CORNER_REGION_RATIO)
    ch = int(meta.height * _CORNER_REGION_RATIO)
    return [
        (0, 0, meta.width, meta.height),
        (0, 0, cw, ch),
        (meta.width - cw, 0, cw, ch),
        (0, meta.height - ch, cw, ch),
        (meta.width - cw, meta.height - ch, cw, ch),
    ]


def _collect_region_bboxes(
    detector,  # noqa: ANN001  -- mediapipe FaceDetector, lazy import
    frames: list[np.ndarray],
    region: tuple[int, int, int, int],
) -> list[tuple[float, float, float, float]]:
    """주어진 region을 frames에서 crop해 BlazeFace로 face bbox 추출. 원본 좌표계로 변환해 반환.

    multi-face 영상은 가정 외라 confidence 가장 높은 1개만 채택.
    """
    import mediapipe as mp

    sx, sy, sw, sh = region
    bboxes: list[tuple[float, float, float, float]] = []
    for frame in frames:
        rgb = cv2.cvtColor(frame[sy : sy + sh, sx : sx + sw], cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        if not result.detections:
            continue
        best = max(
            result.detections,
            key=lambda d: d.categories[0].score if d.categories else 0.0,
        )
        box = best.bounding_box
        bboxes.append((box.origin_x + sx, box.origin_y + sy, box.width, box.height))
    return bboxes


def _aggregate_roi(
    bboxes: list[tuple[float, float, float, float]], meta: _VideoMeta
) -> _ROI:
    """검출된 face bbox들의 평균 중심 + 최대 크기 × ROI_MARGIN_FACTOR로 최종 ROI 산출."""
    arr = np.array(bboxes, dtype=np.float64)
    cx = (arr[:, 0] + arr[:, 2] / 2.0).mean()
    cy = (arr[:, 1] + arr[:, 3] / 2.0).mean()
    bw = arr[:, 2].max() * ROI_MARGIN_FACTOR
    bh = arr[:, 3].max() * ROI_MARGIN_FACTOR
    rx = max(0, int(cx - bw / 2.0))
    ry = max(0, int(cy - bh / 2.0))
    return _ROI(
        x=rx,
        y=ry,
        w=min(meta.width - rx, int(bw)),
        h=min(meta.height - ry, int(bh)),
    )


def _detect_webcam_roi(video_path: str) -> _ROI | None:
    """첫 ROI_DETECTION_WINDOW_SEC 구간에서 BlazeFace로 face bbox를 모아 안정 ROI를 결정.

    강의 영상은 슬라이드가 화면 대부분이라 화자 얼굴이 전체의 1~5%만 차지하면 BlazeFace
    다운스케일에 얼굴이 사라져 검출 실패. 전체 → 4코너 순으로 search region을 좁히며 시도해
    첫 검출 region에서 break.

    None 반환 = 4코너 폴백까지 모두 실패 (영상에 화자 얼굴이 없거나 비표준 위치).
    """
    frames: list[np.ndarray] = []
    with open_capture(video_path) as cap:
        meta = _read_video_meta(cap)
        max_frames = min(int(ROI_DETECTION_WINDOW_SEC * meta.fps), meta.total_frames)
        for frame_idx in range(max_frames):
            if not cap.grab():
                break
            if frame_idx % meta.sample_step != 0:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            frames.append(frame)

    if not frames:
        return None

    detector = _get_face_detector()
    for region in _search_regions(meta):
        bboxes = _collect_region_bboxes(detector, frames, region)
        if bboxes:
            return _aggregate_roi(bboxes, meta)
    return None


def sample_video_pose(video_path: str) -> list[PoseSample]:
    """영상 → 5fps 프레임 단위 head pose 측정 리스트. ROI 추정 실패 시 빈 list."""
    # BlazeFace로 ROI 자동 추정 — 4코너 폴백까지 실패면 FaceLandmarker는 BlazeFace보다
    # 큰 얼굴을 요구하므로 의미 있는 결과를 못 냄 → 즉시 빈 list로 빠른 종료.
    roi = _detect_webcam_roi(video_path)
    if roi is None:
        return []

    # heavy import는 노드 진입 시점에만. 그래프 빌드/임포트 비용 회피.
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    samples: list[PoseSample] = []
    with open_capture(video_path) as cap:
        meta = _read_video_meta(cap)
        cam_w, cam_h = roi.w, roi.h

        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=str(
                    _ensure_model(
                        _LANDMARKER_MODEL_URL,
                        _LANDMARKER_MODEL_PATH,
                        _LANDMARKER_MODEL_SHA256,
                    )
                )
            ),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
            # grab+retrieve로 5fps 샘플링: skip 프레임은 디코딩 없이 grab만 호출.
            # cap.read()로 전수 디코딩하면 80%가 버려져 비용이 약 5배 차이 (mp4 GOP 의존
            # 부정확성 회피 위해 cap.set(POS_FRAMES) 대신 순차 grab 사용).
            for frame_idx in range(meta.total_frames):
                if not cap.grab():
                    break
                if frame_idx % meta.sample_step != 0:
                    continue

                ok, frame = cap.retrieve()
                if not ok:
                    break

                cropped = frame[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
                t = frame_idx / meta.fps
                rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, int(t * 1000))
                if not result.face_landmarks:
                    continue

                lms = result.face_landmarks[0]
                points_2d = np.array(
                    [(lms[i].x * cam_w, lms[i].y * cam_h) for i in _LANDMARK_IDS],
                    dtype=np.float64,
                )
                pose = _solve_head_pose(points_2d, cam_w, cam_h)
                if pose is None:
                    continue

                yaw, pitch = pose
                samples.append(PoseSample(t=t, yaw=yaw, pitch=pitch))

    return samples


def subtract_baseline(
    samples: list[PoseSample],
) -> tuple[list[PoseSample], float, float]:
    """영상 전체 yaw/pitch median을 정면 baseline으로 차감해 (samples, by, bp) 반환.

    median은 outlier에 robust해 짧은 시선 이탈이 baseline 추정을 오염시키지 않는다.
    강사가 영상 내내 시선 이탈 중인 비정상 케이스는 강의 도메인 가정상 무시.
    baseline 값도 반환 — 호출자가 median 재계산 안 하도록.
    """
    if not samples:
        return [], 0.0, 0.0
    yaws = np.array([s.yaw for s in samples])
    pitches = np.array([s.pitch for s in samples])
    by, bp = float(np.median(yaws)), float(np.median(pitches))
    centered = [PoseSample(t=s.t, yaw=s.yaw - by, pitch=s.pitch - bp) for s in samples]
    return centered, by, bp


def _detect_gaze_sync(video_path: str) -> list[GazeEvent]:
    samples, _, _ = subtract_baseline(sample_video_pose(video_path))
    return samples_to_events(samples)


async def detect_gaze_events(video_path: str) -> list[GazeEvent]:
    """영상 → 시선 이탈 이벤트 리스트.

    그래프가 lecture 카테고리에서만 이 노드를 호출한다. MediaPipe·OpenCV는 sync·CPU
    bound이라 to_thread로 분리해 이벤트 루프 차단 방지.
    """
    return await asyncio.to_thread(_detect_gaze_sync, video_path)
