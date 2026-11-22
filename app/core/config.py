from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    gemini_embedding_model: str = "models/gemini-embedding-001"
    gemini_request_timeout: float = Field(default=45.0, gt=0, le=120)

    chroma_url: str = ""
    chroma_collection: str = Field(default="brands_clippers_kb", min_length=3, max_length=63)
    chroma_persist_directory: str = "./chroma_db"

    ai_service_port: int = 8001
    # Authentication must fail closed until the Express signing secret is supplied.
    jwt_secret: str = ""
    jwt_algorithm: Literal["HS256"] = "HS256"
    # Internal review/intelligence routes remain unavailable until configured.
    ai_service_api_key: str = ""
    backend_url: str = "http://localhost:5000"
    frontend_origins: str = "http://localhost:3000"
    database_url: str = ""
    backend_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    chat_rate_limit: int = Field(default=30, gt=0, le=300)
    chat_rate_window_seconds: int = Field(default=60, gt=0, le=3600)
    max_message_chars: int = Field(default=4000, gt=0, le=4000)
    max_history_messages: int = Field(default=12, gt=0, le=40)
    max_history_chars: int = Field(default=12000, gt=0, le=50000)

    model_config = SettingsConfigDict(
        env_file=str(_ENV_PATH) if _ENV_PATH.exists() else ".env",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def allowed_origins(self) -> list[str]:
        origins = [origin.strip() for origin in self.frontend_origins.split(",") if origin.strip()]
        if not origins or "*" in origins:
            raise ValueError("FRONTEND_ORIGINS must contain explicit origins; wildcards are not allowed")
        return origins


@lru_cache()
def get_settings() -> Settings:
    return Settings()
