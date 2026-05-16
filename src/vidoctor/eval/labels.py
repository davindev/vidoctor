"""골든셋 라벨 CSV 로드.

CSV 스키마: start,end,dimension,kind,note
- start/end: 초 단위 (소수점 1자리 권장)
- dimension: filler / cps / dead_zone / gaze / content_gap
- kind: cps에만 too_fast / too_slow (그 외 차원은 빈 값)
- note: 자유 텍스트 (라벨러 메모, 평가에는 영향 없음)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import cast

from pydantic import BaseModel

from vidoctor.graph.state import Dimension


class GoldenLabel(BaseModel):
    start: float
    end: float
    dimension: Dimension
    kind: str | None = None
    note: str = ""


_REQUIRED_COLUMNS = frozenset({"start", "end", "dimension"})
_KNOWN_COLUMNS = frozenset({"start", "end", "dimension", "kind", "note"})


def load_labels(csv_path: Path | str) -> list[GoldenLabel]:
    """CSV를 list[GoldenLabel]로 파싱한다.

    헤더 검증: 필수 컬럼(start/end/dimension)이 빠지거나 모르는 컬럼이 있으면
    ValueError로 즉시 abort — 비싼 ASR 호출 전에 입력 사양 실패. start나
    dimension이 빈 행은 skip. dimension Literal 검증은 Pydantic이 수행.
    """
    path = Path(csv_path)
    labels: list[GoldenLabel] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = frozenset(reader.fieldnames or ())
        missing = _REQUIRED_COLUMNS - fields
        if missing:
            raise ValueError(
                f"라벨 CSV {path.name}에 필수 컬럼이 빠졌습니다: {sorted(missing)} "
                f"(현재 컬럼: {sorted(fields)})"
            )
        unknown = fields - _KNOWN_COLUMNS
        if unknown:
            raise ValueError(
                f"라벨 CSV {path.name}에 알 수 없는 컬럼이 있습니다: {sorted(unknown)} "
                f"(허용 컬럼: {sorted(_KNOWN_COLUMNS)})"
            )
        for row in reader:
            if not row.get("start") or not row.get("dimension"):
                continue
            # kind는 cps에만 채워지는 Optional이라 빈 문자열은 None으로 정규화.
            # note는 항상 str이라 빈값 그대로 보존.
            labels.append(
                GoldenLabel(
                    start=float(row["start"]),
                    end=float(row["end"]),
                    dimension=cast(Dimension, row["dimension"]),
                    kind=row.get("kind") or None,
                    note=row.get("note", ""),
                )
            )
    return labels
