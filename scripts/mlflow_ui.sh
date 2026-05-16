#!/usr/bin/env bash
# MLflow UI launcher — file backend(mlruns/)를 지정 포트로 띄움.
#
# 사용법:
#   bash scripts/mlflow_ui.sh           # 기본 포트 5001
#   PORT=5002 bash scripts/mlflow_ui.sh # 다른 포트
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PORT="${PORT:-5001}"
MLFLOW_BIN=".venv/bin/mlflow"

if [[ ! -x "$MLFLOW_BIN" ]]; then
    echo "❌ $MLFLOW_BIN 없음 — uv sync로 의존성 설치 필요." >&2
    exit 1
fi

if lsof -iTCP:"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "❌ 포트 $PORT 이미 사용 중. PORT=다른포트 bash scripts/mlflow_ui.sh" >&2
    exit 1
fi

exec "$MLFLOW_BIN" ui \
    --host 127.0.0.1 \
    --port "$PORT" \
    --backend-store-uri file:./mlruns
