"""환경변수 기반 설정 — pydantic-settings로 .env 자동 로드."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py(parents[0]) → src/(parents[1]) → repo root(parents[2], .env 위치).
# scripts/·eval/·vision에서 import해서 사용 — 모듈 외부 public 상수.
ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """OpenAI/Supabase/R2/Langfuse/MLflow 자격증명 + 호스트."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: SecretStr

    supabase_url: str
    supabase_service_key: SecretStr

    r2_endpoint: str
    r2_access_key_id: SecretStr
    r2_secret_access_key: SecretStr
    r2_bucket: str

    langfuse_public_key: SecretStr
    langfuse_secret_key: SecretStr
    langfuse_host: str

    # None이면 mlflow native default(`file:./mlruns`) 사용. .env에서 절대경로 sqlite URI를
    # 지정하면 평가 결과·mlflow ui가 동일 store를 보도록 정렬된다.
    mlflow_tracking_uri: str | None = None

    # WhisperX 모델 swap용 dev 옵션. None이면 audio/transcribe.py의 DEFAULT_MODEL_NAME.
    # 기존 VIDOCTOR_WHISPER_MODEL env var 이름 유지 (validation_alias).
    whisper_model: str | None = Field(default=None, validation_alias="VIDOCTOR_WHISPER_MODEL")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """프로세스 수명 동안 1회만 로드. 모든 모듈이 공유."""
    # BaseSettings는 env에서 필드 동적 로드 — pyright가 필수 인자 누락으로 오해.
    return Settings()  # pyright: ignore[reportCallIssue]
