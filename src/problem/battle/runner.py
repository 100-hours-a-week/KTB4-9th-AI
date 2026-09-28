"""배틀 문제 생성 그래프 1회 실행.

새벽 배치가 쓴다. 실패가 배치의 다른 단계를 막지 않도록
예외를 올리지 않고 None으로 알린다.
"""

import logging

from src.core.enums import Trigger
from src.problem.battle.graph import get_battle_graph
from src.problem.battle.state import BattleState
from src.schema.problem import BattleProblem, HiddenTestCase

logger = logging.getLogger(__name__)


async def run_battle_graph(*, trigger: Trigger) -> BattleProblem | None:
    """
    배틀 문제 하나를 생성한다. 검증을 통과하면 finalize가 저장한다.

    카테고리는 그래프가 무작위로 고른다. 폐기되거나 실행 중 오류가 나면
    None을 반환한다.

    Parameters:
        trigger (Trigger): 생성 경로. BATCH로 만든 문제만 새벽에 전송된다

    Returns:
        BattleProblem | None: 생성한 문제. 실패하면 None
    """
    try:
        result = BattleState(
            **await get_battle_graph().ainvoke(BattleState(trigger=trigger))
        )
    except Exception as error:
        logger.warning("배틀 문제 생성 실패: %r", error, exc_info=True)
        return None

    if result.is_discarded:
        logger.info(
            "배틀 문제 폐기 (%s): %s (%s) %s",
            result.category.value if result.category else "-",
            result.discard_reason,
            result.discard_stage,
            result.discard_detail,
        )
        return None

    try:
        return BattleProblem(
            category=result.category,
            problem_title=result.problem_title,
            problem_content=result.problem_content,
            test_cases=[
                HiddenTestCase(input=case.input, output=case.output)
                for case in result.test_cases
            ],
        )
    except Exception as error:
        logger.warning("배틀 문제 변환 실패: %r", error, exc_info=True)
        return None
