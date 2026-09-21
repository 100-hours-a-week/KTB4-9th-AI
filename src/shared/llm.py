import json
import re

from langchain_google_genai import ChatGoogleGenerativeAI

from src.problem.state import LLMConfig
from src.shared.config import get_settings

FENCE_PATTERN = re.compile(r"^```[a-zA-Z]*\n(.*?)\n?```$", re.DOTALL)


class LLMOutputParseError(Exception):
    """모델 응답을 JSON으로 해석할 수 없을 때 발생한다."""


def strip_code_fence(text: str) -> str:
    """
    응답을 감싼 코드펜스(```)를 제거한다.

    Parameters:
        text (str): 모델 응답 원문

    Returns:
        str: 코드펜스를 제거한 본문. 코드펜스가 없으면 앞뒤 공백만 제거
    """
    stripped = text.strip()
    match = FENCE_PATTERN.match(stripped)
    return match.group(1).strip() if match else stripped


def parse_json(text: str) -> dict:
    """
    모델 응답에서 JSON 객체를 꺼낸다.

    코드펜스를 먼저 제거하고, 그래도 해석되지 않으면
    처음 나오는 { 부터 마지막 } 까지만 잘라 다시 시도한다.

    Parameters:
        text (str): 모델 응답 원문

    Returns:
        dict: 해석한 JSON 객체

    Raises:
        LLMOutputParseError: JSON 객체로 해석할 수 없는 경우
    """
    cleaned = strip_code_fence(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise LLMOutputParseError("응답에 JSON 객체가 없음") from None
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as error:
            raise LLMOutputParseError(f"JSON 해석 실패: {error}") from error

    if not isinstance(data, dict):
        raise LLMOutputParseError("응답이 JSON 객체가 아님")
    return data


async def call_llm(prompt: str, cfg: LLMConfig) -> str:
    """
    Gemini에 프롬프트를 보내고 응답 텍스트를 반환한다.

    LLMConfig.top_k는 few-shot 예시 개수이므로 모델 파라미터로 넘기지 않는다.

    Parameters:
        prompt (str): 완성된 프롬프트
        cfg (LLMConfig): 모델명과 temperature를 담은 설정

    Returns:
        str: 모델 응답 텍스트

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

    llm = ChatGoogleGenerativeAI(**options)
    response = await llm.ainvoke(prompt)
    return response.text
