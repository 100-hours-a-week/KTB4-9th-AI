"""지문 모순 검사 — 지문·제약·예시가 서로 맞는지 본다."""

import logging
from typing import Literal

from pydantic import BaseModel, Field

from src.client.llm import LLMConfig, call_llm_structured
from src.core.config import get_settings
from src.core.enums import DiscardReason
from src.problem.render import render_problem
from src.problem.state import GraphState, discard

logger = logging.getLogger(__name__)

PROMPT_VERSION = "check_contradiction/v1"

LOGIC_PROMPT = """당신은 코딩 테스트 문제의 검수자다.

[문제]
{problem}

[판단할 것]
이 문제가 스스로 모순되는지 본다. 풀이 방법을 찾을 필요는 없다.

- problem: 지문 안에서 말이 어긋난다.
  (요구하는 것이 두 가지로 읽힘, 출력 형식이 지문과 다름)
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


async def check_contradiction(state: GraphState) -> dict:
    """
    지문에 모순이 없는지 검사한다.

    Parameters:
        state (GraphState): 문제 필드가 채워진 상태

    Returns:
        dict: 모순 검사 결과
    """

    cfg = LLMConfig(
        model_name=get_settings().llm_models.check_contradiction,
        prompt_version=PROMPT_VERSION,
    )
    updates: dict = {"node_models": {"check_contradiction": cfg}}

    try:
        verdict = await call_llm_structured(
            build_logic_prompt(state), cfg, LogicVerdict
        )
    except Exception as error:
        logger.warning("지문 모순 검사 실패: %s", error, exc_info=True)
        return updates | discard(
            DiscardReason.LLM_ERROR, f"지문 모순 검사 실패: {error}"
        )

    reason = CONTRADICTION_REASONS.get(verdict.contradiction)
    if reason is not None:
        logger.info("지문 모순으로 폐기: %s / %s", reason.value, verdict.reason)
        return updates | discard(reason, verdict.reason)

    return updates
