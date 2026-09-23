"""확정된 문제 저장.

검증을 모두 통과한 문제를 전송 대기 버퍼(generated_problems)에 넣는다.
여기까지 온 문제만 Spring으로 나간다.
"""

import logging
import uuid

from src.core.exception import CosmosError
from src.problem.state import GraphState
from src.schema import Problem
from src.shared.db_client import session_scope
from src.shared.embedding import EMBEDDING_MODEL
from src.shared.repository import (
    GeneratedProblemRepository,
    ProblemEmbeddingRepository,
)

logger = logging.getLogger(__name__)


class ProblemSaveError(CosmosError):
    """확정된 문제를 저장할 수 없다."""


def build_problem(state: GraphState) -> Problem:
    """
    상태를 전송·저장용 Problem 스키마로 옮긴다.

    GraphState와 Problem은 필드 이름이 같다. mode="json"으로 덤프해서
    중첩 모델까지 dict로 만든 뒤 Problem이 다시 검증한다.
    GraphState에만 있는 필드(검증 플래그 등)는 무시된다.

    Parameters:
        state (GraphState): 검증을 모두 통과한 상태

    Returns:
        Problem: 저장·전송할 문제

    Raises:
        ProblemSaveError: 필수 필드가 비어 스키마를 만들 수 없는 경우
    """
    try:
        return Problem.model_validate(state.model_dump(mode="json"))
    except ValueError as error:
        raise ProblemSaveError(f"문제 스키마를 만들 수 없음: {error}") from error


async def index_embedding(
    session, problem_id: uuid.UUID, state: GraphState, problem: Problem
) -> None:
    """
    중복 검사용 임베딩을 색인한다.

    문제 저장과 같은 트랜잭션에서 한다. problem_embeddings가
    generated_problems를 참조하므로 따로 커밋할 수 없다.

    임베딩은 check_duplicate가 만들어 상태에 담아 둔 것을 쓴다. 없으면
    색인을 건너뛴다. 이 문제는 앞으로 중복 검사에 걸리지 않으므로
    조용히 넘기지 않고 ERROR로 남긴다.

    Parameters:
        session: 문제를 저장한 세션
        problem_id (uuid.UUID): 저장된 문제 id
        state (GraphState): algorithm_core와 임베딩이 담긴 상태
        problem (Problem): 저장한 문제
    """
    if not state.algorithm_core_embedding or not state.algorithm_core:
        logger.error(
            "임베딩이 없어 색인하지 못함 (problem_id=%s). "
            "이 문제는 중복 검사에 걸리지 않는다.",
            problem_id,
        )
        return

    await ProblemEmbeddingRepository(session).add(
        problem_id=problem_id,
        category=problem.category,
        algorithm_core=state.algorithm_core,
        embedding=state.algorithm_core_embedding,
        embedding_model=EMBEDDING_MODEL,
    )


async def finalize(state: GraphState) -> dict:
    """
    확정된 문제를 버퍼에 저장한다.

    실패하면 예외를 올린다. 폐기 로그와 달리 여기서 실패하면
    만들어 낸 문제가 사라지는 것이므로, 성공으로 위장해선 안 된다.

    Parameters:
        state (GraphState): 검증을 모두 통과한 상태

    Returns:
        dict: 저장된 문제의 id

    Raises:
        ProblemSaveError: 스키마 변환이나 저장에 실패한 경우
    """
    problem = build_problem(state)

    try:
        async with session_scope() as session:
            row = await GeneratedProblemRepository(session).add(problem)
            problem_id = row.id
            await index_embedding(session, problem_id, state, problem)
    except Exception as error:
        logger.error("문제 저장 실패: %s", error, exc_info=True)
        raise ProblemSaveError(f"문제 저장 실패: {error}") from error

    logger.info(
        "문제 저장: %s / %s / %s",
        problem_id,
        problem.category.value,
        problem.difficulty.value,
    )
    return {"problem_id": problem_id}
