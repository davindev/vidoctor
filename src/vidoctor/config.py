from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]
