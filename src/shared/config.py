from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # gemini_api_key: str
    database_url: str
    # spring_base_url: str
    # judge0_url: str


@lru_cache
def get_settings() -> Settings:
    return Settings()
