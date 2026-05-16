"""5차원 × 2 영상(lecture, vlog) 평가 일괄 실행기.

차원별 *_eval.py를 subprocess로 호출. 각 실행은 자기 ASR/feature 캐시를 재사용하므로
첫 번째 차원만 transcribe 비용 발생, 이후엔 캐시 hit.

vlog는 graph 카테고리에서 gaze·content_gap이 비활성이라 평가에서도 제외 (라벨에
gaze/content_gap 라벨이 없음).

사용법:
    uv run python scripts/eval_all.py --tag stage16
    uv run python scripts/eval_all.py --tag stage16 --videos lecture
    uv run python scripts/eval_all.py --tag stage16 --no-mlflow --force
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from vidoctor.config import ROOT
from vidoctor.graph.state import CATEGORY_DIMENSIONS, Category, Dimension

_VIDEOS: dict[str, Category] = {"lecture": "lecture", "vlog": "vlog"}
_DIMENSION_SCRIPT: dict[Dimension, str] = {
    "filler": "filler_eval.py",
    "cps": "cps_eval.py",
    "dead_zone": "dead_zone_eval.py",
    "gaze": "gaze_eval.py",
    "content_gap": "content_gap_eval.py",
}


def _build_invocation(
    script: str,
    video_name: str,
    category: Category,
    run_name: str,
    *,
    no_mlflow: bool,
    force: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / script),
        str(ROOT / "data" / "golden" / f"{video_name}.mp4"),
        str(ROOT / "data" / "golden" / f"{video_name}_labels.csv"),
    ]
    if script == "dead_zone_eval.py":
        cmd.append(category)
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
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    failures: list[tuple[str, str]] = []
    started = time.perf_counter()
    for video_name in args.videos:
        category = _VIDEOS[video_name]
        active = args.dimensions or list(CATEGORY_DIMENSIONS[category])
        for dim in active:
            if dim not in _DIMENSION_SCRIPT:
                continue
            script = _DIMENSION_SCRIPT[dim]
            run_name = f"{args.tag}_{video_name}"
            cmd = _build_invocation(
                script,
                video_name,
                category,
                run_name,
                no_mlflow=args.no_mlflow,
                force=args.force,
            )
            print(f"\n=== [{dim}] {video_name} → {run_name} ===")
            print("  $ " + " ".join(cmd))
            result = subprocess.run(cmd)
            if result.returncode != 0:
                failures.append((dim, video_name))
                print(f"  ✗ FAILED (exit {result.returncode})")

    elapsed = time.perf_counter() - started
    print(f"\n{'=' * 60}\nelapsed: {elapsed:.1f}s")
    if failures:
        for dim, video in failures:
            print(f"  ✗ {dim} / {video}")
        sys.exit(f"{len(failures)} 차원 실패")
    print("  ✓ 전부 성공")


if __name__ == "__main__":
    main()
