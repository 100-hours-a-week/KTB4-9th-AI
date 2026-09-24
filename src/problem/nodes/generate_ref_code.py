from pydantic import BaseModel, Field

from src.client.llm import LLMConfig, call_llm_structured, strip_code_fence
from src.core.enums import Language
from src.core.exception import LLMOutputParseError
from src.problem.render import render_problem
from src.problem.state import GraphState

MODEL_NAME = "gemini-3.7-flash"
PROMPT_VERSION = "generate_ref_code/v2"
REFERENCE_LANGUAGE = Language.PYTHON

GENERATE_REF_CODE_PROMPT = """당신은 알고리즘 문제 풀이자다.

[문제]
{problem}

[규칙]
- {language}로 푼다. 표준 입력으로 읽고 표준 출력으로 쓴다.
- 표준 라이브러리만 사용한다.
- 실행 제한 {time_limit_ms}ms, {memory_limit_kb}KB 안에 동작해야 한다.
- code에는 설명과 주석 없이 실행 가능한 소스 코드만 넣는다.
"""


class ReferenceCode(BaseModel):
    """검증용 코드 생성 응답 형식."""

    code: str = Field(description="표준 입출력을 사용하는 실행 가능한 소스 코드")


def build_prompt(state: GraphState) -> str:
    """
    검증용 코드 생성 프롬프트를 만든다.

    카테고리와 선정 이유는 넣지 않는다. 풀이자가 지문만으로 풀 수 있어야
    문제가 스스로 완결되어 있다는 근거가 된다.

    Parameters:
        state (GraphState): 문제 필드가 채워진 상태

    Returns:
        str: 완성된 프롬프트
    """
    limit = next(
        lim for lim in state.execution_limits if lim.language == REFERENCE_LANGUAGE
    )
    return GENERATE_REF_CODE_PROMPT.format(
        problem=render_problem(state, include_category=False),
        language=REFERENCE_LANGUAGE,
        time_limit_ms=limit.time_limit_ms,
        memory_limit_kb=limit.memory_limit_kb,
    )


async def generate_ref_code(state: GraphState) -> dict:
    """
    의미 검증에 사용할 레퍼런스 코드를 생성한다.

    Parameters:
        state (GraphState): 정적 검증과 중복 검사를 통과한 상태

    Returns:
        dict: 레퍼런스 코드, 언어, 호출 설정

    Raises:
        LLMOutputParseError: 응답이 형식에 맞지 않거나 코드가 비어 있는 경우
    """
    cfg = LLMConfig(
        model_name=MODEL_NAME,
        prompt_version=PROMPT_VERSION,
        temperature=0.2,
    )

    response = await call_llm_structured(build_prompt(state), cfg, ReferenceCode)
    reference_code = strip_code_fence(response.code)
    if not reference_code:
        raise LLMOutputParseError("검증용 코드가 비어 있음")

    return {
        "reference_code": reference_code,
        "reference_language": REFERENCE_LANGUAGE,
        "node_models": {"generate_ref_code": cfg},
    }
