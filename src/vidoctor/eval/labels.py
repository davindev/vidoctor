"""골든셋 라벨 CSV 로드.

CSV 스키마: start,end,dimension,severity,kind,note
- start/end: 초 단위 (소수점 1자리 권장)
- dimension: filler / cps / dead_zone / gaze / content_gap
- severity: low / mid / high (v1.0은 모든 차원이 mid 통일)
- kind: cps에만 too_fast / too_slow (그 외 차원은 빈 값)
- note: 자유 텍스트 (라벨러 메모, 평가에는 영향 없음)
"""

from __future__ import annotations

import csv
from pathlib import Path

from pydantic import BaseModel

from vidoctor.graph.state import Dimension, Severity


class GoldenLabel(BaseModel):
    start: float
    end: float
    dimension: Dimension
    severity: Severity = "mid"
    kind: str | None = None
    note: str = ""


def load_labels(csv_path: Path | str) -> list[GoldenLabel]:
    """CSV → list[GoldenLabel]. 빈 행/주석은 무시. 파일 없으면 FileNotFoundError.

    dimension·severity Literal 검증은 Pydantic이 수행 — 잘못된 값은 ValidationError.
    """
    path = Path(csv_path)
    labels: list[GoldenLabel] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("start") or not row.get("dimension"):
                continue
            labels.append(
                GoldenLabel(
                    start=float(row["start"]),
                    end=float(row["end"]),
                    dimension=row["dimension"],  # type: ignore[arg-type]
                    severity=row.get("severity") or "mid",  # type: ignore[arg-type]
                    kind=row.get("kind") or None,
                    note=row.get("note", ""),
                )
            )
    return labels
