"""중복 검사.

algorithm_core를 임베딩해 같은 카테고리의 기존 문제와 비교한다.
이야기 설정과 입출력 형식이 달라도 같은 알고리즘이면 걸러야 한다.
"""

import logging

from src.client.embedding import EMBEDDING_MODEL, embed_text
from src.core.enums import Category, DiscardReason
from src.db.repository import ProblemEmbeddingRepository
from src.db.session import session_scope
from src.problem.state import GraphState, discard

logger = logging.getLogger(__name__)

# 이 값 이상이면 중복으로 본다.
# gemini-embedding-2(768차원)로 문제 122개, 폐기 57건을 실측한 값:
#   같은 카테고리의 서로 다른 문제 (434쌍)  중앙 0.718, p95 0.815
#   다른 카테고리                (6947쌍) 중앙 0.649, p95 0.723
#   중복으로 걸러진 문제           (57건)   최소 0.852, 중앙 0.896
# 0.70~0.85 구간에 61%가 몰려 있어 빈 골짜기가 없다. 경계는 데이터가 정해주지 않는다.
# 폐기 57건 중 0.85~0.87 구간 10건을 표본 검토한 결과 자료구조만 같고
# 계산 대상이 다른 오탐이라 0.87로 올렸다.
# 재조정할 때는 problem_embeddings가 아니라 discarded_problems를 봐야 한다.
# 색인에는 임계값을 통과한 문제만 남아 최댓값이 기준선 바로 아래에서 잘린다.
DUPLICATE_SIMILARITY = 0.87

# 비교해 볼 이웃 수. 가장 비슷한 하나로 판정하지만 로그에 몇 개를 남긴다.
NEIGHBOR_COUNT = 3


async def check_duplicate(state: GraphState) -> dict:
    """
    같은 카테고리에 비슷한 문제가 이미 있는지 본다.

    임베딩은 상태에 담아 finalize가 색인에 다시 쓴다. 두 번 부르지 않는다.

    Parameters:
        state (GraphState): generate_problem 결과가 채워진 상태

    Returns:
        dict: 신규면 임베딩. 중복이거나 실패하면 폐기 정보
    """
    if not state.algorithm_core:
        return discard(DiscardReason.EMPTY_FIELD, "핵심 풀이가 비어 중복 검사 불가")
    if not state.category:
        return discard(DiscardReason.EMPTY_FIELD, "카테고리가 비어 중복 검사 불가")

    try:
        category = Category(state.category)
    except ValueError:
        return discard(
            DiscardReason.EMPTY_FIELD, f"알 수 없는 카테고리: {state.category}"
        )

    try:
        embedding = await embed_text(state.algorithm_core)
        async with session_scope() as session:
            neighbors = await ProblemEmbeddingRepository(session).find_similar(
                category=category,
                embedding=embedding,
                embedding_model=EMBEDDING_MODEL,
                limit=NEIGHBOR_COUNT,
            )
    except Exception as error:
        # 임베딩 호출 실패, DB 장애 모두 여기로 온다.
        # 확인하지 못한 문제를 저장하지 않는 쪽을 택한다.
        logger.warning("중복 검사 실패: %s", error, exc_info=True)
        return discard(DiscardReason.LLM_ERROR, f"중복 검사 실패: {error}")

    if neighbors:
        logger.info(
            "가장 비슷한 기존 문제: %s",
            ", ".join(f"{n.problem_id} {n.similarity:.3f}" for n in neighbors),
        )

    closest = neighbors[0] if neighbors else None
    if closest is not None and closest.similarity >= DUPLICATE_SIMILARITY:
        return discard(
            DiscardReason.DUPLICATE,
            f"유사도 {closest.similarity:.3f} (기준 {DUPLICATE_SIMILARITY}) "
            f"- 기존 문제 {closest.problem_id}: {closest.algorithm_core}",
        )

    return {"algorithm_core_embedding": embedding}
