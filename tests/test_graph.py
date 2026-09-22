import pytest

from src.enum import Difficulty, DiscardReason
from src.problem.graph import PARALLEL_NODES, build_graph
from src.problem.nodes import stubs
from src.problem.state import GraphState, discard

INPUT = {"requested_difficulty": "LV2", "requested_category": "DP"}

# 그래프 테스트는 실제 LLM을 호출하지 않도록 LLM 노드를 임시 구현으로 고정한다.
OFFLINE_NODES = {
    "generate_problem": stubs.generate_problem,
    "generate_ref_code": stubs.generate_ref_code,
}


async def run(overrides: dict | None = None) -> GraphState:
    """
    그래프를 실행하고 결과를 GraphState로 반환한다.

    ainvoke 결과에는 노드가 값을 쓴 필드만 담기므로,
    GraphState로 감싸 나머지 필드를 기본값으로 채운다.
    """
    graph = build_graph({**OFFLINE_NODES, **(overrides or {})})
    result = await graph.ainvoke(INPUT)
    return GraphState(**result)


@pytest.mark.asyncio
async def test_정상_흐름이면_병렬_노드까지_실행된다():
    state = await run()

    assert not state.is_discarded
    assert state.is_semantically_valid
    assert set(state.node_models) == set(PARALLEL_NODES)


@pytest.mark.asyncio
async def test_정적검증_실패면_폐기로_끝나고_병렬_노드는_안_돈다():
    async def wrong_difficulty(state: GraphState) -> dict:
        result = await stubs.generate_problem(state)
        result["difficulty"] = Difficulty.LV5
        return result

    state = await run({"generate_problem": wrong_difficulty})

    assert state.discard_reason == DiscardReason.MISMATCH_LABEL
    assert state.node_models == {}


@pytest.mark.asyncio
async def test_중복이면_폐기로_끝난다():
    async def always_duplicate(state: GraphState) -> dict:
        return {
            "is_duplicated": True,
            **discard(DiscardReason.DUPLICATE, "유사 문제 존재"),
        }

    state = await run({"check_duplicate": always_duplicate})

    assert state.discard_reason == DiscardReason.DUPLICATE
    assert state.reference_code is None


@pytest.mark.asyncio
async def test_의미검증이_계속_실패하면_3회_재시도_후_폐기한다():
    async def always_fail(state: GraphState) -> dict:
        attempt = state.semantic_validation_attempt
        if attempt >= state.semantic_validation_max_attempt:
            return discard(DiscardReason.MAX_ATTEMPT_EXCEEDED, "재시도 초과")
        return {"semantic_validation_attempt": attempt + 1}

    state = await run({"semantic_validate": always_fail})

    assert state.discard_reason == DiscardReason.MAX_ATTEMPT_EXCEEDED
    assert state.semantic_validation_attempt == 3
    assert state.node_models == {}
