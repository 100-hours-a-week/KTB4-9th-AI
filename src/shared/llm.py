import re

from langchain_core.exceptions import OutputParserException
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from src.core.config import get_settings
from src.core.exception import LLMOutputParseError
from src.problem.state import LLMConfig

FENCE_PATTERN = re.compile(r"^```[a-zA-Z]*\n(.*?)\n?```$", re.DOTALL)

__all__ = [
    "LLMOutputParseError",
    "build_chat_model",
    "call_llm_structured",
    "strip_code_fence",
]


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

    Parameters:
        prompt (str): 완성된 프롬프트
        cfg (LLMConfig): 모델 설정
        schema (type[T]): 응답 형식을 정의한 Pydantic 클래스

    Returns:
        T: schema 형식으로 검증된 응답 객체

    Raises:
        LLMOutputParseError: 응답이 형식에 맞지 않거나 비어 있는 경우
    """
    structured = build_chat_model(cfg).with_structured_output(schema)
    try:
        result = await structured.ainvoke(prompt)
    except (OutputParserException, ValueError) as error:
        raise LLMOutputParseError(f"구조화 응답 해석 실패: {error}") from error

    if not isinstance(result, schema):
        raise LLMOutputParseError("구조화 응답이 비어 있음")
    return result
