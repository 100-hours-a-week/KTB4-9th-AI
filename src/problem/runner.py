"""문제 생성 그래프 1회 실행.

데일리와 새벽 배치처럼 여러 문제를 한꺼번에 만드는 쪽이 쓴다.
한 문제의 실패가 나머지를 막지 않도록 예외를 올리지 않고 None으로 알린다.
"""

import logging

from src.core.enums import Category, Difficulty, ProblemPurpose, Trigger
from src.problem.graph import get_graph
from src.problem.nodes.finalize import build_problem
from src.problem.state import GraphState
from src.schema.problem import Problem

logger = logging.getLogger(__name__)


async def run_problem_graph(
    difficulty: Difficulty,
    category: Category,
    *,
    trigger: Trigger,
    purpose: ProblemPurpose,
) -> Problem | None:
    """
    문제 하나를 생성한다. 검증을 통과하면 finalize가 버퍼에 저장한다.

    폐기되거나 실행 중 오류가 나면 None을 반환한다.

    Parameters:
        difficulty (Difficulty): 요청 난이도
        category (Category): 요청 카테고리. RANDOM이면 그래프가 고른다
        trigger (Trigger): 생성 경로. BATCH로 만든 문제만 새벽에 전송된다
        purpose (ProblemPurpose): 용도. 전송할 Spring 엔드포인트가 갈린다

    Returns:
        Problem | None: 생성한 문제. 실패하면 None
    """
    label = f"{purpose.value} {category.value}/{difficulty.value}"
    state = GraphState(
        requested_difficulty=difficulty,
        requested_category=category,
        trigger=trigger,
        purpose=purpose,
    )
    try:
        result = GraphState(**await get_graph().ainvoke(state))
    except Exception as error:
        logger.warning("문제 생성 실패 (%s): %r", label, error, exc_info=True)
        return None

    if result.is_discarded:
        logger.info(
            "문제 폐기 (%s): %s (%s)",
            label,
            result.discard_reason,
            result.discard_stage,
        )
        return None

    try:
        return build_problem(result)
    except Exception as error:
        logger.warning("문제 변환 실패 (%s): %r", label, error, exc_info=True)
        return None
