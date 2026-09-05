from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    gemini_api_key: str = ""
    chroma_url: str = "http://localhost:8000"
    chroma_collection: str = "brands_clippers_kb"
    chroma_persist_directory: str = "./chroma_db"
    ai_service_port: int = 8001
    jwt_secret: str = "supersecretkey_payperview_ai_2026"
    jwt_algorithm: str = "HS256"

    model_config = {"env_file": ".env", "case_sensitive": False}



@lru_cache()
def get_settings() -> Settings:
    return Settings()
