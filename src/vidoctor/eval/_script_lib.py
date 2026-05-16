"""평가 스크립트 공용 유틸 — argparse · 로깅 · ASR 캐시 · MLflow.

5개 차원 *_eval.py가 공유하는 헬퍼: 공통 인자 파서, run_name 접두 로깅, 입력 파일
검증, 평가 결과 JSON 안전 저장, ASR 캐시 로드, MLflow 기록.
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
from vidoctor.config import ROOT, get_settings
from vidoctor.graph.state import Word

_log = logging.getLogger(__name__)

_GOLDEN_DIR = ROOT / "data" / "golden"
_INPUTS_DIR = _GOLDEN_DIR / "inputs"
_EVAL_DUMPS_DIR = _GOLDEN_DIR / "eval_dumps"


def configure_eval_logging(run_name: str, *, level: int = logging.INFO) -> None:
    """모든 로그에 `[run_name] [logger]` 접두를 박아 동시 실행 시 grep으로 분리."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(f"[{run_name}] [%(name)s] %(message)s")
    )
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level)


def _existing_file(raw: str) -> Path:
    """존재하는 파일 경로만 통과시키는 argparse type validator."""
    p = Path(raw)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"파일을 찾을 수 없습니다: {raw}")
    if not p.is_file():
        raise argparse.ArgumentTypeError(f"파일이 아닙니다: {raw}")
    return p


def build_eval_parser(description: str) -> argparse.ArgumentParser:
    """공통 인자가 등록된 argparse 파서를 반환한다."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("video_path", type=_existing_file)
    parser.add_argument("labels_csv", type=_existing_file)
    parser.add_argument(
        "--run-name",
        required=True,
        help="MLflow run 이름 (예: baseline_lecture, stage1_lecture)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="transcript 및 feature 캐시 무시 후 재추출",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="MLflow 기록 생략 (로컬 디버깅용)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 dump 파일 덮어쓰기 허용 (기본: 충돌 시 중단)",
    )
    return parser


def write_eval_dump(out_path: Path, data: dict, *, force: bool) -> None:
    """평가 결과를 JSON으로 저장한다 (기존 파일 + force=False면 FileExistsError)."""
    if out_path.exists() and not force:
        raise FileExistsError(
            f"파일이 이미 존재합니다: {out_path}. "
            "--force로 덮어쓰거나 다른 --run-name을 사용하세요."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def model_tag() -> str:
    """WhisperX 모델별 캐시 키를 반환한다."""
    model = os.environ.get("VIDOCTOR_WHISPER_MODEL")
    if not model:
        return "default"
    return Path(model).name.replace("/", "_") or "default"


def transcript_cache_path(video_path: Path) -> Path:
    """모델 태그가 붙은 ASR 캐시 파일 경로를 반환한다."""
    return _INPUTS_DIR / f"transcript_{video_path.stem}_{model_tag()}.json"


def eval_dump_path(dimension: str, video_stem: str, run_name: str) -> Path:
    """차원별 평가 결과 JSON 경로를 반환한다."""
    return _EVAL_DUMPS_DIR / dimension / f"{video_stem}_{run_name}.json"


def load_or_transcribe(video_path: Path, no_cache: bool) -> list[Word]:
    """ASR 결과를 캐시에서 로드하거나, 없으면 WhisperX로 추출 후 캐시한다."""
    cache = transcript_cache_path(video_path)
    if cache.exists() and not no_cache:
        _log.info("캐시된 transcript 로드: %s", cache.name)
        data = json.loads(cache.read_text())
        return [Word(**w) for w in data]

    _log.info("전사 중: %s", video_path.name)
    words, _ = asyncio.run(transcribe_video(str(video_path)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps([w.model_dump() for w in words], ensure_ascii=False, indent=2)
    )
    _log.info("  → %d 단어 (캐시 저장: %s)", len(words), cache.name)
    return words


def log_mlflow_run(
    experiment: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    """params와 metrics를 MLflow run으로 기록한다 (실패는 warning 후 무시)."""
    settings = get_settings()
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    try:
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
    except Exception as e:  # noqa: BLE001
        _log.warning("MLflow 기록 실패: %s", e)
        return
    _log.info("  → MLflow run 기록: %s / %s", experiment, run_name)
