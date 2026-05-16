"""평가 스크립트 공용 유틸 — transcript cache + MLflow 셋업.

`scripts/{filler,cps,content_gap}_eval.py`가 모두 똑같이 복사해 쓰던 헬퍼 3종(`_model_tag`,
`_transcript_cache_path`, `_load_or_transcribe`) + 5개 스크립트가 반복하던 MLflow 셋업
boilerplate를 한 곳에 모은다. 평가 결과 dump 구조는 차원별 의도가 다르므로 각자 유지.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import mlflow

from vidoctor.audio.transcribe import transcribe_video
from vidoctor.config import get_settings
from vidoctor.graph.state import Word


def configure_eval_logging(run_name: str, *, level: int = logging.INFO) -> None:
    """평가 스크립트 진입 1회 호출. 모든 로그 라인에 `[run_name] [logger]` 접두를
    붙여 동시 실행 시 출력 분리 가능. JSON 포매터(production용)과 분리 — 평가는
    interactive 가독성 우선.

    grep `[stage11_lecture]`로 한 run의 로그만, `[scripts.cps_eval]`로 한 스크립트의
    로그만 골라낼 수 있다.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(f"[{run_name}] [%(name)s] %(message)s")
    )
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level)

_log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[3]
_GOLDEN_DIR = _ROOT / "data" / "golden"
_INPUTS_DIR = _GOLDEN_DIR / "inputs"
_EVAL_DUMPS_DIR = _GOLDEN_DIR / "eval_dumps"


def build_eval_parser(description: str) -> argparse.ArgumentParser:
    """5개 차원 평가 스크립트 공통 인자(video_path · labels_csv · --run-name ·
    --no-cache · --no-mlflow · --force). 차원별 추가 인자는 호출자가 parser에 더
    add_argument."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("video_path", type=_existing_file)
    parser.add_argument("labels_csv", type=_existing_file)
    parser.add_argument(
        "--run-name",
        required=True,
        help="MLflow run name (예: baseline_lecture, stage1_lecture)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="transcript/feature cache 무시하고 재추출",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="MLflow 로그 생략 (디버그용)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 평가 dump JSON 덮어쓰기 허용 (default: 충돌 시 abort)",
    )
    return parser


def _existing_file(raw: str) -> Path:
    """argparse type validator — 존재하는 파일만 통과. 평가 시작 직전에 즉시 실패."""
    p = Path(raw)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"file not found: {raw}")
    if not p.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {raw}")
    return p


def write_eval_dump(out_path: Path, data: dict, *, force: bool) -> None:
    """평가 결과 JSON dump. 기존 파일 존재 시 force가 없으면 abort — 의도치 않은
    덮어쓰기로 옛 detected/labels 본문이 소실되는 사고 방지."""
    if out_path.exists() and not force:
        raise FileExistsError(
            f"이미 존재합니다: {out_path}. --force로 덮어쓰거나 다른 --run-name 사용."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def model_tag() -> str:
    """현재 환경의 WhisperX 모델 식별자 — cache 키로 사용.

    환경변수 미설정이면 "default", 경로면 basename, HF id면 마지막 segment.
    """
    model = os.environ.get("VIDOCTOR_WHISPER_MODEL")
    if not model:
        return "default"
    return Path(model).name.replace("/", "_") or "default"


def transcript_cache_path(video_path: Path) -> Path:
    return _INPUTS_DIR / f"transcript_{video_path.stem}_{model_tag()}.json"


def eval_dump_path(dimension: str, video_stem: str, run_name: str) -> Path:
    """차원별 평가 dump 경로. eval_dumps/{dim}/{video}_{run_name}.json.

    상위 폴더는 write_eval_dump가 mkdir(parents=True). 차원 prefix는 폴더가
    분리하므로 파일명에서 제거 — `eval_dumps/cps/lecture_stage11.json`.
    """
    return _EVAL_DUMPS_DIR / dimension / f"{video_stem}_{run_name}.json"


def load_or_transcribe(video_path: Path, no_cache: bool) -> list[Word]:
    """캐시된 transcript JSON이 있으면 그걸 로드, 없으면 WhisperX 호출 + 캐시 작성."""
    cache = transcript_cache_path(video_path)
    if cache.exists() and not no_cache:
        _log.info("loading cached transcript: %s", cache.name)
        data = json.loads(cache.read_text())
        return [Word(**w) for w in data]

    _log.info("transcribing %s...", video_path.name)
    words, _ = asyncio.run(transcribe_video(str(video_path)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps([w.model_dump() for w in words], ensure_ascii=False, indent=2)
    )
    _log.info("  → %d words (cached → %s)", len(words), cache.name)
    return words


def log_mlflow_run(
    experiment: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    """평가 스크립트 5곳에서 반복하던 MLflow 셋업·로그 한 묶음."""
    settings = get_settings()
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
    _log.info("  → mlflow run logged (%s / %s)", experiment, run_name)
