"""Global configuration for Code Risk Agent."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_api_base: str = Field(
        default="https://api.openai.com/v1", alias="OPENAI_API_BASE"
    )
    model_name: str = Field(default="gpt-4o-mini", alias="MODEL_NAME")
    embedding_model: str = Field(
        default="text-embedding-3-small", alias="EMBEDDING_MODEL"
    )
    db_path: str = Field(default="data/risk_agent.db", alias="DB_PATH")
    max_reflection_rounds: int = 3
    risk_threshold: float = 0.6

    class Config:
        env_file = str(PROJECT_ROOT / ".env")
        env_file_encoding = "utf-8"

    @property
    def db_url(self) -> str:
        p = Path(self.db_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)


settings = Settings()
