from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr
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

    # 로컬 모델 서버 (vLLM, OpenAI 호환).
    # 평가와 배치 생성에서 쓴다. URL에 /v1은 붙이지 않는다
    vllm_url: str | None = None
    vllm_api_key: str | None = None

    # 배치 생성 노드가 부를 vLLM 모델 이름. 비어 있으면 배치도 Gemini를 쓴다
    batch_local_model: str | None = None
    # 재시도해도 실패하면 Gemini로 다시 만든다
    batch_local_fallback: bool = True
    batch_local_timeout_s: float = 180  # 시도 한 번의 상한
    # True/False면 enable_thinking으로 보낸다. None이면 모델 기본값
    batch_local_thinking: bool | None = None

    # 관리자 API 인증 토큰. 비어 있으면 관리자 API를 전부 거절한다
    admin_token: SecretStr | None = None

    # Langfuse (LLM 관측성 및 프롬프트 트레이싱).
    # 비어 있으면 트레이스 수집을 건너뛰고 정상 동작한다.
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://jp.cloud.langfuse.com"

    backend_url: str = "http://localhost:8080"

    judge0_url: str = "http://localhost:2358"

    mock_llm: bool = False

    batch_enabled: bool = False
    batch_timezone: str = "Asia/Seoul"
    batch_concurrency: int = 5

    log_level: str = "INFO"

    # AWS SSM Parameter Store 모델 버전 설정 (Dev: us-west-2, Prod: ap-northeast-2)
    ssm_model_version_param: str | None = None
    aws_region: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
