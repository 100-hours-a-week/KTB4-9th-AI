"""폐기 로그 저장.

폐기된 문제는 버리지만 사유는 남긴다. 사유별 집계가 있어야
어느 검증이 얼마나 걸러내는지, 프롬프트를 고쳐야 하는지 알 수 있다.
"""

import logging
from typing import Any

from src.enum import Category
from src.problem.state import GraphState
from src.shared.db_client import session_scope
from src.shared.repository import DiscardedProblemRepository

logger = logging.getLogger(__name__)


def build_discard_record(state: GraphState) -> dict[str, Any] | None:
    """
    폐기 로그 한 행으로 만들 값을 모은다.

    사유가 없으면 None을 돌려준다. 폐기로 왔는데 사유가 없다는 것은
    어느 노드가 discard()를 거치지 않고 상태만 바꿨다는 뜻이다.

    Parameters:
        state (GraphState): 폐기로 끝난 상태

    Returns:
        dict[str, Any] | None: 리포지토리에 넘길 인자. 기록할 수 없으면 None
    """
    if state.discard_reason is None:
        return None

    try:
        category = Category(state.requested_category)
    except ValueError:
        logger.warning(
            "폐기 로그의 카테고리를 해석할 수 없음: %r", state.requested_category
        )
        return None

    return {
        "requested_category": category,
        "requested_difficulty": state.requested_difficulty,
        # build_graph가 노드 이름을 채운다. 비어 있으면 어디서 걸렸는지 모른다.
        "stage": state.discard_stage or "unknown",
        "reason": state.discard_reason,
        "detail": state.discard_detail,
        # 폐기된 문제를 다시 들여다볼 수 있게 상태를 남긴다.
        # 임베딩은 숫자 768개라 로그만 부풀리고 읽을 값이 없어 뺀다.
        "raw_payload": state.model_dump(
            mode="json", exclude={"algorithm_core_embedding"}
        ),
    }


async def discard_problem(state: GraphState) -> dict:
    """
    폐기 사유를 남긴다.

    저장에 실패해도 예외를 올리지 않는다. 문제는 이미 폐기됐고,
    로그를 못 남긴 것 때문에 요청 전체를 실패로 만들 이유가 없다.

    Parameters:
        state (GraphState): 폐기로 끝난 상태

    Returns:
        dict: 상태를 바꾸지 않으므로 빈 dict
    """
    record = build_discard_record(state)
    if record is None:
        logger.error(
            "폐기 사유가 없어 로그를 남기지 못함 (stage=%s)", state.discard_stage
        )
        return {}

    logger.info(
        "문제 폐기: %s / %s - %s",
        record["stage"],
        record["reason"].value,
        record["detail"],
    )
    try:
        async with session_scope() as session:
            await DiscardedProblemRepository(session).add(**record)
    except Exception as error:
        # 폐기 통계가 비는 것은 문제이므로 스택까지 남긴다
        logger.error("폐기 로그 저장 실패: %s", error, exc_info=True)
    return {}
