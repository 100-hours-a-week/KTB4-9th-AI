from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = ".env"

load_dotenv(ENV_FILE, override=False)


class LLMModels(BaseModel):
    generate_problem: str = "gemini-3.8-flash"
    check_contradiction: str = "gemini-3.8-flash"
    generate_ref_code: str = "gemini-3.5-flash-lite"
    generate_nl_keyword: str = "gemini-3.5-flash-lite"
    generate_testcase: str = "gemini-3.5-flash-lite"
    generate_solution_code: str = "gemini-3.5-flash-lite"
    generate_battle_problem: str = "gemini-3.5-flash-lite"
    evaluation_nl_solution: str = "gemini-3.5-flash-lite"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE, extra="ignore", env_nested_delimiter="__"
    )

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

    llm_models: LLMModels = LLMModels()

    # 평가용 로컬 모델 서버 (Colab vLLM + ngrok). 운영 경로에서는 쓰지 않는다
    vllm_url: str | None = None
    vllm_api_key: str | None = None

    backend_url: str = "http://localhost:8080"

    judge0_url: str = "http://localhost:2358"

    mock_llm: bool = False

    batch_enabled: bool = False
    batch_timezone: str = "Asia/Seoul"
    batch_concurrency: int = 5

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
