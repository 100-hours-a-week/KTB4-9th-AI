import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import TYPE_CHECKING

from langchain_core.exceptions import OutputParserException
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, ValidationError

from src.core.config import get_settings
from src.core.exception import LLMOutputParseError

if TYPE_CHECKING:
    from openai import AsyncOpenAI

FENCE_PATTERN = re.compile(r"^```[a-zA-Z]*\n(.*?)\n?```$", re.DOTALL)

__all__ = [
    "build_chat_model",
    "call_llm_structured",
    "current_local_model",
    "strip_code_fence",
    "use_local_model",
    "LLMConfig",
    "LocalModel",
]


class LLMConfig(BaseModel):
    model_name: str | None = None
    model_version: str | None = None
    prompt_version: str | None = None
    temperature: float | None = None
    top_k: int | None = 3


class LocalModel(BaseModel):
    """vLLM 같은 OpenAI 호환 서버에 띄운 모델. 평가에서만 쓴다.

    샘플링 값을 비워 두면 서버가 모델의 generation_config 기본값을 쓴다.
    """

    model_name: str  # 서버에 요청할 이름 (base repo id 또는 LoRA 어댑터 이름)
    chat_template_kwargs: dict | None = None  # 예: {"enable_thinking": False}
    temperature: float | None = None
    timeout_s: float = 300


# 설정돼 있으면 이 컨텍스트 안의 call_llm_structured가 Gemini 대신 이 모델을 부른다
_local_model: ContextVar[LocalModel | None] = ContextVar("local_model", default=None)


@contextmanager
def use_local_model(model: LocalModel) -> Iterator[None]:
    """
    블록 안의 LLM 호출을 로컬 모델 서버로 보낸다.

    평가에서 노드 하나만 바꿔 끼울 때 그 노드 실행을 이 블록으로 감싼다.
    ContextVar라 동시에 도는 다른 요청(다른 task)에는 영향이 없다.

    Parameters:
        model (LocalModel): 요청을 받을 모델
    """
    token = _local_model.set(model)
    try:
        yield
    finally:
        _local_model.reset(token)


def current_local_model() -> LocalModel | None:
    """지금 컨텍스트에서 쓰는 로컬 모델. 없으면 None (Gemini)."""
    return _local_model.get()


def strip_code_fence(text: str) -> str:
    """
    코드 문자열을 감싼 코드펜스(```)나 인라인 코드 표시(`)를 제거한다.

    Parameters:
        text (str): 모델이 생성한 코드 문자열

    Returns:
        str: 감싼 표시를 제거한 본문. 없으면 앞뒤 공백만 제거
    """
    stripped = text.strip()
    match = FENCE_PATTERN.match(stripped)
    if match:
        return match.group(1).strip()
    if len(stripped) >= 2 and stripped.startswith("`") and stripped.endswith("`"):
        return stripped.strip("`").strip()
    return stripped


def build_chat_model(cfg: LLMConfig) -> ChatGoogleGenerativeAI:
    """
    설정값으로 Gemini 채팅 모델 객체를 만든다.

    LLMConfig.top_k는 few-shot 예시 개수이므로 모델 파라미터로 넘기지 않는다.

    Parameters:
        cfg (LLMConfig): 모델명과 temperature를 담은 설정

    Returns:
        ChatGoogleGenerativeAI: 호출 가능한 모델 객체

    Raises:
        RuntimeError: API 키가 설정되지 않은 경우
        ValueError: 모델명이 지정되지 않은 경우
    """
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")
    if not cfg.model_name:
        raise ValueError("LLMConfig.model_name이 비어 있습니다.")

    options = {"model": cfg.model_name, "google_api_key": settings.gemini_api_key}
    if cfg.temperature is not None:
        options["temperature"] = cfg.temperature
    return ChatGoogleGenerativeAI(**options)


async def call_llm_structured[T: BaseModel](
    prompt: str, cfg: LLMConfig, schema: type[T]
) -> T:
    """
    응답 형식을 지정해 Gemini를 호출하고 그 형식의 객체로 반환한다.

    모든 노드의 LLM 호출은 이 함수를 사용한다.
    노드마다 모델 설정(cfg)과 응답 형식(schema)을 다르게 넘긴다.
    use_local_model 블록 안이면 Gemini 대신 그 로컬 모델을 부른다.

    Parameters:
        prompt (str): 완성된 프롬프트
        cfg (LLMConfig): 모델 설정
        schema (type[T]): 응답 형식을 정의한 Pydantic 클래스

    Returns:
        T: schema 형식으로 검증된 응답 객체

    Raises:
        LLMOutputParseError: 응답이 형식에 맞지 않거나 비어 있는 경우
    """
    local = _local_model.get()
    if local is not None:
        return await call_local_structured(prompt, local, schema)

    structured = build_chat_model(cfg).with_structured_output(schema)
    try:
        result = await structured.ainvoke(prompt)
    except (OutputParserException, ValueError) as error:
        raise LLMOutputParseError(f"구조화 응답 해석 실패: {error}") from error

    if not isinstance(result, schema):
        raise LLMOutputParseError("구조화 응답이 비어 있음")
    return result


@lru_cache
def get_local_client() -> AsyncOpenAI:
    """
    로컬 모델 서버(vLLM)의 OpenAI 호환 클라이언트.

    LangSmith 추적이 켜져 있으면 호출이 그래프 트레이스 아래에 기록된다.
    평가에서만 쓰므로 import를 여기서 해 운영 경로의 import에 영향을 주지 않는다.

    Raises:
        RuntimeError: VLLM_URL 또는 VLLM_API_KEY가 설정되지 않은 경우
    """
    settings = get_settings()
    if not settings.vllm_url or not settings.vllm_api_key:
        raise RuntimeError("VLLM_URL, VLLM_API_KEY가 설정되지 않았습니다.")

    from langsmith.wrappers import wrap_openai
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=f"{settings.vllm_url.rstrip('/')}/v1",
        api_key=settings.vllm_api_key,
    )
    return wrap_openai(client)


async def call_local_structured[T: BaseModel](
    prompt: str, model: LocalModel, schema: type[T]
) -> T:
    """
    로컬 모델 서버에 응답 형식을 지정해 요청하고 그 형식의 객체로 반환한다.

    Gemini 경로와 같은 프롬프트를 user 메시지 하나로 보낸다.
    response_format은 출력 형식만 강제하고 모델에게 스키마를 보여 주지는 않는다.

    Parameters:
        prompt (str): 완성된 프롬프트
        model (LocalModel): 요청을 받을 모델
        schema (type[T]): 응답 형식을 정의한 Pydantic 클래스

    Returns:
        T: schema 형식으로 검증된 응답 객체

    Raises:
        LLMOutputParseError: 응답이 형식에 맞지 않거나 비어 있는 경우
    """
    options = {}
    if model.temperature is not None:
        options["temperature"] = model.temperature
    if model.chat_template_kwargs:
        options["extra_body"] = {"chat_template_kwargs": model.chat_template_kwargs}

    response = await get_local_client().chat.completions.create(
        model=model.model_name,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema.model_json_schema(),
            },
        },
        timeout=model.timeout_s,
        **options,
    )
    choice = response.choices[0]
    text = choice.message.content or ""
    if not text.strip():
        raise LLMOutputParseError("구조화 응답이 비어 있음")
    try:
        return schema.model_validate_json(text)
    except ValidationError as error:
        # finish_reason이 length면 출력이 잘려 JSON이 끝나지 않은 것이다
        raise LLMOutputParseError(
            f"구조화 응답 해석 실패 (finish_reason={choice.finish_reason}): {error}"
        ) from error
