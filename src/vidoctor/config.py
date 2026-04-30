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
    supabase_anon_key: SecretStr
    supabase_service_key: SecretStr

    langfuse_public_key: SecretStr
    langfuse_secret_key: SecretStr
    langfuse_host: str = "https://cloud.langfuse.com"

    huggingface_token: SecretStr | None = None


def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]
