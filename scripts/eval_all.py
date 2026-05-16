"""5차원 × 2 영상(lecture, vlog) 평가 일괄 실행기.

차원별 *_eval.py를 subprocess로 호출. 각 실행은 자기 ASR/feature 캐시를 재사용하므로
첫 번째 차원만 transcribe 비용 발생, 이후엔 캐시 hit.

vlog는 graph 카테고리에서 gaze·content_gap이 비활성이라 평가에서도 제외 (라벨에
gaze/content_gap 라벨이 없음).

사용법:
    uv run python scripts/eval_all.py --tag stage16
    uv run python scripts/eval_all.py --tag stage16 --videos lecture
    uv run python scripts/eval_all.py --tag stage16 --dimensions filler cps
    uv run python scripts/eval_all.py --tag stage16 --no-mlflow --force
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time

from vidoctor.config import ROOT
from vidoctor.eval._script_lib import configure_eval_logging
from vidoctor.graph.state import CATEGORY_DIMENSIONS, Category, Dimension

_log = logging.getLogger(__name__)

_VIDEOS: dict[str, Category] = {"lecture": "lecture", "vlog": "vlog"}
_DIMENSION_SCRIPT: dict[Dimension, str] = {
    "filler": "filler_eval.py",
    "cps": "cps_eval.py",
    "dead_zone": "dead_zone_eval.py",
    "gaze": "gaze_eval.py",
    "content_gap": "content_gap_eval.py",
}
# 차원별 category 인자 전달 방식. dead_zone은 positional, content_gap은 --category.
_DIMS_WITH_POSITIONAL_CATEGORY: frozenset[Dimension] = frozenset({"dead_zone"})
_DIMS_WITH_FLAG_CATEGORY: frozenset[Dimension] = frozenset({"content_gap"})


def _build_invocation(
    dim: Dimension,
    video_name: str,
    category: Category,
    run_name: str,
    *,
    no_mlflow: bool,
    force: bool,
) -> list[str]:
    """차원별 *_eval.py subprocess 호출에 쓸 cmd 리스트를 만든다."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / _DIMENSION_SCRIPT[dim]),
        str(ROOT / "data" / "golden" / "inputs" / f"{video_name}.mp4"),
        str(ROOT / "data" / "golden" / "labels" / f"{video_name}_labels.csv"),
    ]
    if dim in _DIMS_WITH_POSITIONAL_CATEGORY:
        cmd.append(category)
    if dim in _DIMS_WITH_FLAG_CATEGORY:
        cmd += ["--category", category]
    cmd += ["--run-name", run_name]
    if no_mlflow:
        cmd.append("--no-mlflow")
    if force:
        cmd.append("--force")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="5차원 × 영상 평가 일괄 실행")
    parser.add_argument(
        "--tag",
        required=True,
        help="run name prefix (예: stage16 → run_name이 stage16_lecture/_vlog)",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        choices=list(_VIDEOS),
        default=list(_VIDEOS),
        help=f"실행할 영상 (default: {' '.join(_VIDEOS)})",
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        choices=list(_DIMENSION_SCRIPT),
        default=None,
        help="실행할 차원 (default: 영상 카테고리의 활성 차원)",
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
    args = parser.parse_args()
    configure_eval_logging(args.tag)

    failures: list[tuple[str, str]] = []
    started = time.perf_counter()
    for video_name in args.videos:
        category = _VIDEOS[video_name]
        active = args.dimensions or list(CATEGORY_DIMENSIONS[category])
        for dim in active:
            if dim not in _DIMENSION_SCRIPT:
                continue
            run_name = f"{args.tag}_{video_name}"
            cmd = _build_invocation(
                dim,
                video_name,
                category,
                run_name,
                no_mlflow=args.no_mlflow,
                force=args.force,
            )
            _log.info("=== [%s] %s → %s ===", dim, video_name, run_name)
            _log.info("$ %s", " ".join(cmd))
            result = subprocess.run(cmd)
            if result.returncode != 0:
                failures.append((dim, video_name))
                _log.error("✗ FAILED (exit %d)", result.returncode)

    elapsed = time.perf_counter() - started
    _log.info("=" * 60)
    _log.info("elapsed: %.1fs", elapsed)
    if failures:
        for dim, video in failures:
            _log.error("✗ %s / %s", dim, video)
        sys.exit(f"{len(failures)} 차원 실패")
    _log.info("✓ 전부 성공")


if __name__ == "__main__":
    main()
