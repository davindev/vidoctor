#!/usr/bin/env bash
# MLflow UI launcher — file backend(mlruns/)를 5001 포트로 띄움.
# 사용법: bash scripts/mlflow_ui.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec .venv/bin/mlflow ui \
    --host 127.0.0.1 \
    --port 5001 \
    --backend-store-uri file:./mlruns
