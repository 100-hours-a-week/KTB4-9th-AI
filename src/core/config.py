from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = ".env"

load_dotenv(ENV_FILE, override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = "cosmos"
    db_password: str = ""
    db_name: str = "cosmos"

    db_echo: bool = False
    db_pool_size: int = 5
    db_max_overflow: int = 5

    @property
    def db_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    gemini_api_key: str | None = None

    backend_url: str = "http://localhost:8080"

    judge0_url: str = "http://localhost:2358"

    batch_enabled: bool = False
    batch_timezone: str = "Asia/Seoul"
    batch_concurrency: int = 5

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
