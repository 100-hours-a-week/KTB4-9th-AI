import asyncio

import pytest
from pydantic_core import to_jsonable_python

from evals.generate import GenerationRow
from evals.requests import EvalRequest
from evals.score import (
    build_score_graph,
    generation_sha,
    is_infra_failure,
    score_one,
    select_scoring,
)
from src.core.enums import Category, Difficulty, DiscardReason
from src.core.exception import LLMOutputParseError
from src.problem.graph import PARALLEL_NODES
from src.problem.state import GraphState, discard
from tests import fake_nodes
from tests.fake_nodes import OFFLINE_NODES

REQUEST = EvalRequest(
    id="v1-0001", difficulty=Difficulty.LV2, category=Category.DP, seed=1
)


# 1단계가 저장하는 형태(JSON)의 정상 문제. 가짜 generate_problem의 출력이다
BASE_OUTPUT = to_jsonable_python(
    asyncio.run(
        fake_nodes.generate_problem(
            GraphState(requested_difficulty="LV2", requested_category="DP")
        )
    )
)


def generated_output(**changes) -> dict:
    return BASE_OUTPUT | changes


def generation(output: dict | None = None, status: str = "ok") -> GenerationRow:
    return GenerationRow.model_validate(
        {
            "request_id": REQUEST.id,
            "label": "test",
            "model": "qwen3-4b-ft",
            "difficulty": "LV2",
            "category": "DP",
            "status": status,
            "seconds": 30.0,
            "output": output if status == "ok" else None,
            "started_at": "2026-10-08T00:00:00Z",
        }
    )


def forbidden(name: str):
    async def node(state: GraphState) -> dict:
        raise AssertionError(f"{name}가 실행됐다")

    return node


def offline(**overrides) -> dict:
    """LLM·judge0를 부르지 않는 노드. 정적 검증만 실 구현으로 돈다."""
    return {
        **OFFLINE_NODES,
        # 멈춰야 하는 병렬 생성 노드와 DB 노드는 실행되면 실패한다
        **{name: forbidden(name) for name in PARALLEL_NODES},
        "finalize": forbidden("finalize"),
        "discard_problem": forbidden("discard_problem"),
        "check_duplicate": forbidden("check_duplicate"),
        **overrides,
    }


def fail_with(reason: DiscardReason):
    async def node(state: GraphState) -> dict:
        return discard(reason, "테스트 폐기")

    return node


async def score(output: dict, nodes: dict | None = None):
    graph = build_score_graph(nodes or offline())
    return await score_one(graph, REQUEST, generation(output), label="test")


# ── 채점 ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_의미_검증까지_통과하면_병렬_노드_전에_멈춘다():
    row = await score(generated_output())

    assert row.status == "passed", row.error or row.discard_detail
    assert row.node_calls["generate_ref_code"] == 1
    assert row.node_calls["semantic_validate"] == 1
    assert not set(PARALLEL_NODES) & set(row.node_calls), "병렬 노드까지 갔다"
    assert row.model == "qwen3-4b-ft"


@pytest.mark.asyncio
async def test_저장된_문제를_정적_검증에_그대로_넣는다():
    """JSON으로 저장한 문제가 그래프 안에서 다시 모델로 바뀌어 검증돼야 한다."""
    row = await score(generated_output(difficulty="LV5"))

    assert row.status == "discarded"
    assert row.discard_stage == "static_validate"
    assert row.discard_reason == DiscardReason.MISMATCH_LABEL.value
    assert "generate_ref_code" not in row.node_calls


@pytest.mark.asyncio
async def test_의미_검증에서_폐기되면_사유를_남긴다():
    nodes = offline(semantic_validate=fail_with(DiscardReason.EXAMPLE_MISMATCH))

    row = await score(generated_output(), nodes)

    assert row.status == "discarded"
    assert row.discard_stage == "semantic_validate"
    assert row.discard_reason == DiscardReason.EXAMPLE_MISMATCH.value
    assert not is_infra_failure(row), "문제 탓인 폐기다"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason", [DiscardReason.LLM_ERROR, DiscardReason.EXECUTOR_ERROR]
)
async def test_채점_쪽_장애로_폐기되면_인프라_실패로_본다(reason):
    row = await score(generated_output(), offline(semantic_validate=fail_with(reason)))

    assert row.status == "discarded"
    assert is_infra_failure(row)


@pytest.mark.asyncio
async def test_예외가_나면_에러로_남기고_노드를_기록한다():
    async def broken(state: GraphState) -> dict:
        raise LLMOutputParseError("검증용 코드가 비어 있음")

    row = await score(generated_output(), offline(generate_ref_code=broken))

    assert row.status == "error"
    assert row.error_node == "generate_ref_code"
    assert is_infra_failure(row), "레퍼런스 코드는 Gemini가 만든다"


@pytest.mark.asyncio
async def test_동시에_채점해도_문제가_섞이지_않는다():
    graph = build_score_graph(offline())
    good = generation(generated_output())
    bad = generation(generated_output(difficulty="LV5"))

    passed, discarded = await asyncio.gather(
        score_one(graph, REQUEST, good, label="test"),
        score_one(graph, REQUEST, bad, label="test"),
    )

    assert passed.status == "passed"
    assert discarded.discard_reason == DiscardReason.MISMATCH_LABEL.value


# ── 채점 대상 고르기 ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_생성에_실패한_요청은_채점하지_않는다():
    generations = {REQUEST.id: generation(status="error")}

    assert select_scoring([REQUEST], generations, {}, retry_errors=True) == []


@pytest.mark.asyncio
async def test_채점한_뒤_다시_생성되면_다시_채점한다():
    old = generated_output()
    scored = await score(old)
    generations = {REQUEST.id: generation(generated_output(problem_title="새 제목"))}

    assert scored.generation_sha == generation_sha(old)
    assert select_scoring(
        [REQUEST], generations, {REQUEST.id: scored}, retry_errors=False
    ) == [REQUEST]


@pytest.mark.asyncio
async def test_인프라_실패는_retry_errors일_때만_다시_채점한다():
    output = generated_output()
    infra = await score(
        output, offline(semantic_validate=fail_with(DiscardReason.LLM_ERROR))
    )
    generations = {REQUEST.id: generation(output)}
    scores = {REQUEST.id: infra}

    assert select_scoring([REQUEST], generations, scores, retry_errors=False) == []
    assert select_scoring([REQUEST], generations, scores, retry_errors=True) == [
        REQUEST
    ]


@pytest.mark.asyncio
async def test_통과한_문제는_다시_채점하지_않는다():
    output = generated_output()
    passed = await score(output)

    assert (
        select_scoring(
            [REQUEST],
            {REQUEST.id: generation(output)},
            {REQUEST.id: passed},
            retry_errors=True,
        )
        == []
    )
