"""온디맨드로 만든 문제의 전달 처리.

Spring이 응답을 기다리다 연결을 끊으면 만든 문제가 전달되지 않는다.
그런 문제는 저장 API로 따로 보내 재고로 쓰게 한다.
"""

import logging
import uuid

import httpx

from src.client.backend import save_problems
from src.db.repository import GeneratedProblemRepository
from src.db.session import session_scope
from src.schema.problem import Problem

logger = logging.getLogger(__name__)


async def mark_delivered(problem_id: uuid.UUID) -> None:
    """
    Spring에 전달된 문제를 보낸 것으로 표시한다.

    실패해도 예외를 올리지 않는다. 응답은 이미 만들어졌기 때문이다.

    Parameters:
        problem_id (uuid.UUID): 우리 DB의 문제 id
    """
    try:
        async with session_scope() as session:
            await GeneratedProblemRepository(session).mark_sent([problem_id])
    except Exception as error:
        logger.warning("전달 표시 실패 (id=%s): %r", problem_id, error)


async def save_undelivered(problem_id: uuid.UUID, problem: Problem) -> None:
    """
    응답으로 전달하지 못한 문제를 Spring 저장 API로 보낸다.

    실패해도 예외를 올리지 않는다. 문제는 우리 DB에 남아 있다.

    Parameters:
        problem_id (uuid.UUID): 우리 DB의 문제 id
        problem (Problem): 보낼 문제
    """
    try:
        await save_problems([problem])
    except httpx.HTTPError as error:
        logger.error(
            "전달하지 못한 문제를 Spring에 저장하지 못함 (id=%s): %r",
            problem_id,
            error,
        )
        return

    await mark_delivered(problem_id)
    logger.info("전달하지 못한 문제를 Spring에 저장함 (id=%s)", problem_id)
