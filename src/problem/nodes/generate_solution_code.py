import asyncio
import logging

from pydantic import BaseModel, Field

from src.core.exception import LLMOutputParseError
from src.enum import DiscardReason, Language
from src.problem.nodes.generate_ref_code import REFERENCE_LANGUAGE
from src.problem.render import render_problem
from src.problem.state import (
    ExecutionLimit,
    GraphState,
    HintComment,
    LLMConfig,
    SolutionCode,
    discard,
)
from src.shared.llm import call_llm_structured, strip_code_fence

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-3.7-flash"
PROMPT_VERSION = "generate_solution_code/v1"

HINT_RULE = """- hint에는 이 언어로 풀 때 막히기 쉬운 지점을 한두 문장으로 적는다.
  코드를 줄줄이 설명하지 말고, 자료구조 선택이나 입출력 처리처럼
  실수하기 쉬운 곳을 짚는다."""

TRANSLATE_PROMPT = """당신은 알고리즘 문제 풀이자다.

[문제]
{problem}

[검증된 파이썬 풀이]
{reference_code}

[규칙]
- 위 풀이와 같은 알고리즘으로 {language} 코드를 작성한다.
- 표준 입력으로 읽고 표준 출력으로 쓴다. 표준 라이브러리만 사용한다.
- 실행 제한 {time_limit_ms}ms, {memory_limit_kb}KB 안에 동작해야 한다.
- code에는 설명과 주석 없이 실행 가능한 소스 코드만 넣는다.
{hint_rule}
"""

HINT_PROMPT = """당신은 알고리즘 문제 풀이자다.

[문제]
{problem}

[{language} 풀이]
{reference_code}

[규칙]
{hint_rule}
"""


class LanguageSolution(BaseModel):
    """언어별 정답 코드 응답 형식."""

    code: str = Field(description="표준 입출력을 사용하는 실행 가능한 소스 코드")
    hint: str = Field(description="이 언어로 풀 때 주의할 점 한두 문장")


class SolutionHint(BaseModel):
    """힌트만 받는 응답 형식. 레퍼런스 언어는 코드를 다시 만들지 않는다."""

    hint: str = Field(description="이 언어로 풀 때 주의할 점 한두 문장")


def build_translate_prompt(
    state: GraphState, language: Language, limit: ExecutionLimit
) -> str:
    """
    다른 언어로 옮기는 프롬프트를 만든다.

    검증된 파이썬 풀이를 기준으로 준다. 언어마다 새로 풀게 하면
    알고리즘이 갈려 같은 문제의 정답 코드끼리 어긋난다.

    Parameters:
        state (GraphState): reference_code가 채워진 상태
        language (Language): 옮길 대상 언어
        limit (ExecutionLimit): 그 언어의 실행 제한

    Returns:
        str: 완성된 프롬프트
    """
    return TRANSLATE_PROMPT.format(
        problem=render_problem(state, include_category=False),
        reference_code=state.reference_code,
        language=language.value,
        time_limit_ms=limit.time_limit_ms,
        memory_limit_kb=limit.memory_limit_kb,
        hint_rule=HINT_RULE,
    )


def build_hint_prompt(state: GraphState) -> str:
    """레퍼런스 언어용. 코드는 그대로 쓰고 힌트만 받는다."""
    return HINT_PROMPT.format(
        problem=render_problem(state, include_category=False),
        language=REFERENCE_LANGUAGE.value,
        reference_code=state.reference_code,
        hint_rule=HINT_RULE,
    )


async def solve_language(
    state: GraphState, language: Language, limit: ExecutionLimit, cfg: LLMConfig
) -> tuple[str, str]:
    """
    한 언어의 정답 코드와 힌트를 만든다.

    레퍼런스 언어는 이미 검증된 코드가 있으므로 다시 만들지 않는다.

    Parameters:
        state (GraphState): reference_code가 채워진 상태
        language (Language): 대상 언어
        limit (ExecutionLimit): 그 언어의 실행 제한
        cfg (LLMConfig): 모델 설정

    Returns:
        tuple[str, str]: (정답 코드, 힌트)

    Raises:
        LLMOutputParseError: 응답이 형식에 맞지 않거나 코드가 비어 있는 경우
    """
    if language is REFERENCE_LANGUAGE:
        response = await call_llm_structured(
            build_hint_prompt(state), cfg, SolutionHint
        )
        return state.reference_code or "", response.hint

    response = await call_llm_structured(
        build_translate_prompt(state, language, limit), cfg, LanguageSolution
    )
    code = strip_code_fence(response.code)
    if not code:
        raise LLMOutputParseError(f"{language.value} 정답 코드가 비어 있음")
    return code, response.hint


async def generate_solution_code(state: GraphState) -> dict:
    """
    언어별 정답 코드와 힌트 주석을 만든다.

    네 언어를 동시에 호출하고, 하나라도 실패하면 폐기한다.
    실행 제한이 네 언어 모두 있으므로 정답 코드도 네 언어를 다 채워야 한다.

    병렬 노드이므로 예외를 올리지 않는다. 올리면 그래프 전체가 죽는다.

    Parameters:
        state (GraphState): 의미 검증을 통과한 상태

    Returns:
        dict: 언어별 정답 코드, 힌트, 호출 설정. 실패 시 폐기 정보
    """
    cfg = LLMConfig(
        model_name=MODEL_NAME,
        prompt_version=PROMPT_VERSION,
        temperature=0.2,
    )

    if not state.reference_code:
        return discard(DiscardReason.EMPTY_FIELD, "검증용 코드가 비었음")

    limits = {limit.language: limit for limit in state.execution_limits}
    languages = list(Language)
    missing = [lang.value for lang in languages if lang not in limits]
    if missing:
        return discard(
            DiscardReason.EMPTY_FIELD, f"실행 제한 없음: {', '.join(missing)}"
        )

    results = await asyncio.gather(
        *(solve_language(state, lang, limits[lang], cfg) for lang in languages),
        return_exceptions=True,
    )

    failures: list[str] = []
    for language, result in zip(languages, results, strict=True):
        if isinstance(result, BaseException):
            # 키 누락, 타임아웃, 레이트 리밋도 여기로 온다
            logger.warning(
                "%s 정답 코드 생성 실패: %s", language.value, result, exc_info=result
            )
            failures.append(f"{language.value}: {result}")
    if failures:
        return discard(
            DiscardReason.LLM_ERROR, f"정답 코드 생성 실패: {'; '.join(failures)}"
        )

    solved = list(zip(languages, results, strict=True))
    return {
        "solution_codes": [
            SolutionCode(language=language, content=code)
            for language, (code, _) in solved
        ],
        "hint_comments": [
            HintComment(language=language, content=hint)
            for language, (_, hint) in solved
        ],
        "node_models": {"generate_solution_code": cfg},
    }
