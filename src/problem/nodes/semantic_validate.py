"""의미 검증.

두 단계로 나뉘고, 순서가 비용을 결정한다.

1. 지문 모순 검사 (LLM 1회) — 지문·제약·예시가 서로 맞는지 본다.
2. 실행 기반 검사 — 레퍼런스 코드를 공개 예시에 돌려 출력을 맞춰 본다.

모순 검사를 먼저 하는 이유: 지문이 모순이면 레퍼런스 코드를 몇 번 다시 만들어도
통과하지 못한다. 재시도 한도까지 코드 생성과 실행을 반복하는 비용을 아낀다.
그리고 모순 여부는 코드를 다시 만들어도 달라지지 않으므로 첫 시도에만 검사한다.
"""

import logging
from typing import Literal

from pydantic import BaseModel, Field

from src.client.judge0 import run_code
from src.client.llm import LLMConfig, call_llm_structured
from src.core.enums import DiscardReason
from src.problem.nodes.generate_ref_code import REFERENCE_LANGUAGE
from src.problem.render import render_problem
from src.problem.state import (
    ExecutionLimit,
    GraphState,
    discard,
    find_limit,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-3.7-flash"
PROMPT_VERSION = "semantic_validate/v1"

LOGIC_PROMPT = """당신은 코딩 테스트 문제의 검수자다.

[문제]
{problem}

[판단할 것]
이 문제가 스스로 모순되는지 본다. 풀이 방법을 찾을 필요는 없다.

- problem: 지문 안에서 말이 어긋난다.
  (요구하는 것이 두 가지로 읽힘, 출력 형식이 지문과 다름,
   예시 출력이 지문의 요구와 맞지 않음)
- constraint: 지문과 제약 조건이 어긋난다.
  (제약 범위에서는 지문이 요구하는 상황이 생길 수 없음,
   예시가 제약을 벗어남, 제약끼리 동시에 만족할 수 없음)
- none: 위 어느 것도 아니다.

[규칙]
- 확실한 모순만 집는다. 애매하면 none으로 둔다.
- 문제가 어렵다는 것은 모순이 아니다.
- reason에는 판단 근거를 한두 문장으로 적는다.
"""

CONTRADICTION_DESCRIPTION = (
    "none=모순 없음, problem=지문 자체가 모순, constraint=지문과 제약이 모순"
)

CONTRADICTION_REASONS = {
    "problem": DiscardReason.PROBLEM_CONTRADICTION,
    "constraint": DiscardReason.CONSTRAINT_CONTRADICTION,
}


class LogicVerdict(BaseModel):
    """지문 모순 검사 응답 형식."""

    contradiction: Literal["none", "problem", "constraint"] = Field(
        description=CONTRADICTION_DESCRIPTION
    )
    reason: str = Field(description="판단 근거 한두 문장")


def build_logic_prompt(state: GraphState) -> str:
    """
    모순 검사 프롬프트를 만든다.

    카테고리와 선정 이유를 넣는다. 검수자는 문제의 분류를 아는 상태다.

    Parameters:
        state (GraphState): 문제 필드가 채워진 상태

    Returns:
        str: 완성된 프롬프트
    """
    return LOGIC_PROMPT.format(problem=render_problem(state, include_category=True))


def normalize_output(text: str) -> str:
    """
    출력을 비교할 수 있는 모양으로 만든다.

    줄 끝 공백과 끝의 빈 줄을 무시한다. 채점기가 보통 이렇게 비교한다.

    Parameters:
        text (str): 표준 출력 또는 기대 출력

    Returns:
        str: 비교용으로 정리한 문자열
    """
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


async def check_examples(
    state: GraphState, limit: ExecutionLimit
) -> tuple[DiscardReason, str] | None:
    """
    공개 예시마다 레퍼런스 코드를 돌려 출력을 맞춰 본다.

    Parameters:
        state (GraphState): reference_code와 공개 예시가 채워진 상태
        limit (ExecutionLimit): 레퍼런스 언어의 실행 제한

    Returns:
        tuple[DiscardReason, str] | None: 어긋난 첫 예시의 사유. 모두 맞으면 None
    """
    for index, example in enumerate(state.problem_examples, start=1):
        result = await run_code(
            state.reference_code or "", REFERENCE_LANGUAGE, example.input, limit
        )
        if not result.succeeded:
            tail = result.stderr.strip().splitlines()
            detail = f" ({tail[-1][:80]})" if tail else ""
            return (
                DiscardReason.REFERENCE_CODE_FAILED,
                f"예시 {index}: {result.status.value}{detail}",
            )

        actual = normalize_output(result.stdout)
        expected = normalize_output(example.output)
        if actual != expected:
            return (
                DiscardReason.EXAMPLE_MISMATCH,
                f"예시 {index}: 기대 {expected!r} / 실행 결과 {actual!r}",
            )
    return None


async def semantic_validate(state: GraphState) -> dict:
    """
    지문이 스스로 맞는지, 레퍼런스 코드가 공개 예시를 재현하는지 검사한다.

    실행 기반 검사가 실패하면 레퍼런스 코드를 다시 만들어 재시도한다
    (route_after_semantic이 generate_ref_code로 되돌린다).
    한도를 다 쓰면 실패한 실제 사유로 폐기한다.

    Parameters:
        state (GraphState): 검증용 코드까지 만들어진 상태

    Returns:
        dict: 검증 결과. 실패 시 재시도 표시 또는 폐기 정보
    """
    attempt = state.semantic_validation_attempt + 1
    updates: dict = {"semantic_validation_attempt": attempt}

    if not state.reference_code:
        return updates | discard(DiscardReason.EMPTY_FIELD, "검증용 코드가 비었음")
    if not state.problem_examples:
        return updates | discard(DiscardReason.EMPTY_FIELD, "공개 예시가 없음")

    limit = find_limit(state.execution_limits, REFERENCE_LANGUAGE)
    if limit is None:
        return updates | discard(
            DiscardReason.EMPTY_FIELD, f"{REFERENCE_LANGUAGE.value} 실행 제한 없음"
        )

    # ---- 1. 모순 검사. 지문은 재시도해도 그대로이므로 첫 시도에만 부른다.
    if state.semantic_validation_attempt == 0:
        cfg = LLMConfig(
            model_name=MODEL_NAME,
            prompt_version=PROMPT_VERSION,
            temperature=0.2,
        )
        updates["node_models"] = {"semantic_validate": cfg}
        try:
            verdict = await call_llm_structured(
                build_logic_prompt(state), cfg, LogicVerdict
            )
        except Exception as error:
            # 응답 형식 오류뿐 아니라 키 누락, 타임아웃, 레이트 리밋도 여기로 온다
            logger.warning("지문 모순 검사 실패: %s", error, exc_info=True)
            return updates | discard(
                DiscardReason.LLM_ERROR, f"지문 모순 검사 실패: {error}"
            )

        reason = CONTRADICTION_REASONS.get(verdict.contradiction)
        if reason is not None:
            # 여기서 끝내면 코드 생성과 실행을 더 하지 않는다
            logger.info("지문 모순으로 폐기: %s / %s", reason.value, verdict.reason)
            return updates | discard(reason, verdict.reason)

    # ---- 2. 실행 기반 검사
    failure = await check_examples(state, limit)
    if failure is None:
        return updates | {"is_semantically_valid": True}

    reason, detail = failure
    if attempt < state.semantic_validation_max_attempt:
        # 레퍼런스 코드를 다시 만들어 본다
        logger.info("의미 검증 재시도 %d회: %s", attempt, detail)
        return updates | {"is_semantically_valid": False}

    return updates | discard(reason, f"{detail} ({attempt}회 시도)")
