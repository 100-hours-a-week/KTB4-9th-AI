import asyncio
import logging
import random

from src.core.enums import Category, Difficulty
from src.problem.graph import get_graph
from src.problem.nodes.finalize import build_problem
from src.problem.state import GraphState
from src.schema.problem import Problem

logger = logging.getLogger(__name__)

DAILY_COUNT = 5
MAX_ROUNDS = 3

# 데일리는 전체 유저 공통이라 어렵지 않은 난이도로 고정한다.
DAILY_DIFFICULTIES = [Difficulty.LV1, Difficulty.LV2]
DAILY_CATEGORIES = [c for c in Category if c != Category.RANDOM]


async def generate_one(category: Category) -> Problem | None:
    """
    주어진 카테고리로 문제 하나를 생성한다. 난이도는 무작위로 고른다.

    폐기되거나 실행 중 오류가 나면 None을 반환한다.
    한 문제의 실패가 나머지 생성을 막지 않도록 예외를 올리지 않는다.

    Parameters:
        category (Category): 생성할 카테고리

    Returns:
        Problem | None: 생성한 문제. 실패하면 None
    """
    state = GraphState(
        requested_difficulty=random.choice(DAILY_DIFFICULTIES),
        requested_category=category,
    )
    try:
        result = GraphState(**await get_graph().ainvoke(state))
    except Exception as error:
        logger.warning("데일리 문제 생성 실패: %r", error, exc_info=True)
        return None

    if result.is_discarded:
        logger.info(
            "데일리 문제 폐기: %s (%s)", result.discard_reason, result.discard_stage
        )
        return None

    try:
        return build_problem(result)
    except Exception as error:
        logger.warning("데일리 문제 변환 실패: %r", error, exc_info=True)
        return None


async def generate_daily_problems() -> list[Problem]:
    """
    데일리 문제를 DAILY_COUNT개까지 생성한다.

    한 회차 안에서는 카테고리가 겹치지 않도록 뽑아 동시에 실행한다.
    부족한 만큼 다시 시도하며, MAX_ROUNDS회를 넘기면 만들어진 만큼 반환한다.

    Returns:
        list[Problem]: 생성한 문제 목록. 최대 DAILY_COUNT개
    """
    problems: list[Problem] = []
    used: set[Category] = set()

    for round_no in range(1, MAX_ROUNDS + 1):
        missing = DAILY_COUNT - len(problems)
        remaining = [c for c in DAILY_CATEGORIES if c not in used]

        # 남은 카테고리가 부족하면 전체에서 다시 고른다.
        if len(remaining) < missing:
            remaining = DAILY_CATEGORIES
            used.clear()

        picked = random.sample(remaining, missing)
        used.update(picked)

        results = await asyncio.gather(*(generate_one(c) for c in picked))
        problems.extend(problem for problem in results if problem is not None)

        if len(problems) >= DAILY_COUNT:
            break

        logger.info(
            "데일리 문제 %d/%d개 생성, %d회차 재시도",
            len(problems),
            DAILY_COUNT,
            round_no,
        )

    return problems
