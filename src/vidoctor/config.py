"""환경변수 기반 설정 — pydantic-settings로 .env 자동 로드."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

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
    whisper_model: str | None = Field(default=None, validation_alias="VIDOCTOR_WHISPER_MODEL")

    # CORS 허용 origin. 환경변수는 콤마 구분 문자열(예: "https://a,https://b"),
    # 코드에서는 변경 불가 튜플로 노출. 기본은 로컬 dev, prod에선 Vercel URL을 콤마로 이어 붙인다.
    # NoDecode: pydantic-settings의 list/tuple env 자동 JSON 파싱을 끄고 raw 문자열을
    # 그대로 validator로 흘려보낸다 (URL은 JSON으로 invalid해서 파싱이 깨졌던 케이스).
    frontend_origins: Annotated[tuple[str, ...], NoDecode] = Field(
        default=("http://localhost:3000", "http://127.0.0.1:3000"),
        validation_alias="VIDOCTOR_FRONTEND_ORIGINS",
    )

    @field_validator("frontend_origins", mode="before")
    @classmethod
    def _split_origins(cls, raw: object) -> object:
        if isinstance(raw, str):
            return tuple(o.strip() for o in raw.split(",") if o.strip())
        return raw

    # 전체 사용자 합산 일일 분석 한도. IP rate limit이 우회되어도 OpenAI 비용
    # 폭증을 막는 절대 상한 역할.
    daily_quota: int = Field(default=50, validation_alias="VIDOCTOR_DAILY_QUOTA")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """프로세스 수명 동안 1회만 로드. 모든 모듈이 공유."""
    # BaseSettings는 env에서 필드 동적 로드 — pyright가 필수 인자 누락으로 오해.
    return Settings()  # pyright: ignore[reportCallIssue]
