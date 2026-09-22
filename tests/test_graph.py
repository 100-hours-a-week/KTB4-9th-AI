import pytest

from src.enum import Difficulty, DiscardReason
from src.problem.graph import PARALLEL_NODES, build_graph
from src.problem.nodes import stubs
from src.problem.state import (
    GraphState,
    discard,
    keep_discard_flag,
    keep_first_discard,
)

INPUT = {"requested_difficulty": "LV2", "requested_category": "DP"}

# 그래프 테스트는 실제 LLM을 호출하지 않도록 LLM 노드를 임시 구현으로 고정한다.
OFFLINE_NODES = {
    "generate_problem": stubs.generate_problem,
    "generate_ref_code": stubs.generate_ref_code,
    "generate_testcase": stubs.generate_testcase,
    "generate_solution_code": stubs.generate_solution_code,
    "generate_nl_keyword": stubs.generate_nl_keyword,
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


def _fail_with(reason: DiscardReason, detail: str):
    """항상 주어진 사유로 폐기하는 노드를 만든다."""

    async def node(state: GraphState) -> dict:
        return discard(reason, detail)

    return node


def _counting_graph(**overrides):
    """finalize와 discard_problem이 몇 번 불렸는지 셀 수 있는 그래프."""
    calls = {"finalize": 0, "discard_problem": 0}

    async def finalize(state: GraphState) -> dict:
        calls["finalize"] += 1
        return {}

    async def discard_problem(state: GraphState) -> dict:
        calls["discard_problem"] += 1
        return {}

    graph = build_graph(
        overrides={
            **OFFLINE_NODES,
            "finalize": finalize,
            "discard_problem": discard_problem,
            **overrides,
        }
    )
    return graph, calls


def _initial_state() -> GraphState:
    return GraphState(requested_difficulty=Difficulty.LV2, requested_category="DP")


@pytest.mark.asyncio
async def test_parallel_success_reaches_finalize_once() -> None:
    graph, calls = _counting_graph()

    state = GraphState(**await graph.ainvoke(_initial_state()))

    assert calls == {"finalize": 1, "discard_problem": 0}
    assert state.is_discarded is False


@pytest.mark.asyncio
async def test_one_parallel_failure_skips_finalize() -> None:
    """한 노드만 실패해도 불완전한 문제를 저장하지 않는다."""
    graph, calls = _counting_graph(
        generate_testcase=_fail_with(DiscardReason.LLM_ERROR, "테케 실패")
    )

    state = GraphState(**await graph.ainvoke(_initial_state()))

    assert calls == {"finalize": 0, "discard_problem": 1}
    assert state.discard_reason == DiscardReason.LLM_ERROR
    assert state.discard_detail == "테케 실패"


@pytest.mark.asyncio
async def test_concurrent_failures_do_not_crash_the_graph() -> None:
    """리듀서가 없으면 InvalidUpdateError로 그래프 전체가 죽는다."""
    graph, calls = _counting_graph(
        generate_testcase=_fail_with(DiscardReason.LLM_ERROR, "테케 실패"),
        generate_solution_code=_fail_with(
            DiscardReason.REFERENCE_CODE_FAILED, "코드 실패"
        ),
        generate_nl_keyword=_fail_with(DiscardReason.LLM_ERROR, "키워드 실패"),
    )

    state = GraphState(**await graph.ainvoke(_initial_state()))

    assert calls == {"finalize": 0, "discard_problem": 1}
    assert state.is_discarded is True
    # 셋 중 하나의 사유가 남고, reason과 detail은 같은 노드 것으로 짝을 맞춘다
    assert (state.discard_reason, state.discard_detail) in {
        (DiscardReason.LLM_ERROR, "테케 실패"),
        (DiscardReason.REFERENCE_CODE_FAILED, "코드 실패"),
        (DiscardReason.LLM_ERROR, "키워드 실패"),
    }


def test_keep_discard_flag_never_unsets() -> None:
    assert keep_discard_flag(True, False) is True
    assert keep_discard_flag(False, True) is True


def test_keep_first_discard_keeps_the_earlier_value() -> None:
    assert keep_first_discard("먼저", "나중") == "먼저"
    assert keep_first_discard(None, "나중") == "나중"
