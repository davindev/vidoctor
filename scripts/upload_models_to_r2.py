"""KsponSpeech fine-tuned Whisper 모델을 R2에 업로드.

CI(GitHub Actions)에서 빌드 시 모델을 다운로드받기 위한 사전 업로드 스크립트.
1.5GB라 git에 포함하기 어려워 R2에 보관하고 빌드 시 가져오는 방식.

로컬 `.env`의 R2 자격 증명을 사용합니다.

사용법:
    uv run python scripts/upload_models_to_r2.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import boto3
from botocore.config import Config

from vidoctor.config import ROOT, get_settings

_log = logging.getLogger(__name__)

# R2에 저장될 경로 prefix. CI 다운로드 시 같은 경로 사용.
R2_KEY_PREFIX = "models/whisper-ko-ksponspeech-ct2"
LOCAL_MODEL_DIR = ROOT / "models" / "whisper-ko-ksponspeech-ct2"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    settings = get_settings()

    if not LOCAL_MODEL_DIR.exists():
        raise SystemExit(f"로컬 모델 디렉토리 없음: {LOCAL_MODEL_DIR}")

    client = boto3.client(
        "s3",
        endpoint_url=settings.r2_endpoint,
        aws_access_key_id=settings.r2_access_key_id.get_secret_value(),
        aws_secret_access_key=settings.r2_secret_access_key.get_secret_value(),
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )

    for file_path in sorted(LOCAL_MODEL_DIR.iterdir()):
        if not file_path.is_file():
            continue
        key = f"{R2_KEY_PREFIX}/{file_path.name}"
        size_mb = file_path.stat().st_size / 1024 / 1024
        _log.info("업로드 시작: %s (%.1f MB) → r2://%s/%s", file_path.name, size_mb, settings.r2_bucket, key)
        client.upload_file(str(file_path), settings.r2_bucket, key)
        _log.info("업로드 완료: %s", file_path.name)

    _log.info("모든 모델 파일 업로드 완료")


if __name__ == "__main__":
    main()
