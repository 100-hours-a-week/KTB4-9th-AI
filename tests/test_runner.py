import pytest

from src.core.enums import (
    Category,
    Difficulty,
    DiscardReason,
    ProblemPurpose,
    Trigger,
)
from src.problem import runner as module
from src.problem.graph import build_graph
from src.problem.runner import run_problem_graph
from src.problem.state import GraphState, discard
from src.schema.problem import Problem
from tests.fake_nodes import OFFLINE_NODES


def use_graph(monkeypatch: pytest.MonkeyPatch, **overrides) -> None:
    """LLM·DB를 건드리지 않는 그래프로 바꿔 끼운다."""
    graph = build_graph({**OFFLINE_NODES, **overrides})
    monkeypatch.setattr(module, "get_graph", lambda: graph)


async def run(**options) -> Problem | None:
    return await run_problem_graph(
        Difficulty.LV2,
        Category.DP,
        trigger=options.get("trigger", Trigger.BATCH),
        purpose=options.get("purpose", ProblemPurpose.NORMAL),
    )


@pytest.mark.asyncio
async def test_passed_problem_is_returned(monkeypatch) -> None:
    use_graph(monkeypatch)

    problem = await run()

    assert isinstance(problem, Problem)
    assert problem.category is Category.DP
    assert problem.difficulty is Difficulty.LV2


@pytest.mark.asyncio
async def test_request_reaches_the_graph(monkeypatch) -> None:
    seen: list[GraphState] = []

    async def capture(state: GraphState) -> dict:
        seen.append(state)
        return {}

    use_graph(monkeypatch, finalize=capture)

    await run(trigger=Trigger.BATCH, purpose=ProblemPurpose.DAILY)

    state = seen[0]
    assert state.requested_difficulty is Difficulty.LV2
    assert state.requested_category == Category.DP
    assert state.trigger is Trigger.BATCH
    assert state.purpose is ProblemPurpose.DAILY


@pytest.mark.asyncio
async def test_discarded_problem_is_none(monkeypatch) -> None:
    async def always_duplicate(state: GraphState) -> dict:
        return discard(DiscardReason.DUPLICATE, "유사 문제 존재")

    use_graph(monkeypatch, check_duplicate=always_duplicate)

    assert await run() is None


@pytest.mark.asyncio
async def test_graph_error_is_none_not_raised(monkeypatch) -> None:
    """한 문제의 실패가 같이 돌던 나머지 생성을 멈추면 안 된다."""

    class BrokenGraph:
        async def ainvoke(self, state):
            raise RuntimeError("LLM 연결 끊김")

    monkeypatch.setattr(module, "get_graph", lambda: BrokenGraph())

    assert await run() is None
