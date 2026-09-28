import pytest

from src.core.enums import Category, DiscardReason, Trigger
from src.problem.battle import runner as module
from src.problem.battle.graph import build_battle_graph
from src.problem.battle.runner import run_battle_graph
from src.problem.battle.state import BattleState, BattleTestCase, discard
from src.schema.problem import BattleProblem

INPUTS = ["3\n1 2 3", "1\n5", "2\n-1 1"]


async def generated(state: BattleState) -> dict:
    return {
        "category": Category.ARRAY,
        "problem_title": "배열 합",
        "problem_content": "N개의 정수 합을 출력하라",
        "algorithm_core": "배열을 한 번 돌며 합을 구한다",
        "test_cases": [BattleTestCase(input=i) for i in INPUTS],
        "reference_code": "print(1)",
    }


async def filled(state: BattleState) -> dict:
    return {
        "test_cases": [
            BattleTestCase(input=case.input, output="1") for case in state.test_cases
        ]
    }


async def passed(state: BattleState) -> dict:
    return {}


# LLM·judge0·DB를 건드리지 않는 노드
OFFLINE_NODES = {
    "generate_battle_problem": generated,
    "static_validate": passed,
    "fill_outputs": filled,
    "check_duplicate": passed,
    "finalize": passed,
}


def use_graph(monkeypatch: pytest.MonkeyPatch, **overrides) -> None:
    graph = build_battle_graph({**OFFLINE_NODES, **overrides})
    monkeypatch.setattr(module, "get_battle_graph", lambda: graph)


@pytest.mark.asyncio
async def test_passed_problem_is_returned(monkeypatch) -> None:
    use_graph(monkeypatch)

    problem = await run_battle_graph(trigger=Trigger.BATCH)

    assert isinstance(problem, BattleProblem)
    assert problem.category is Category.ARRAY
    assert [case.input for case in problem.test_cases] == INPUTS
    assert all(case.output == "1" for case in problem.test_cases)


@pytest.mark.asyncio
async def test_trigger_reaches_finalize(monkeypatch) -> None:
    """finalize가 trigger를 저장해야 새벽 전송이 배치 문제만 고른다."""
    seen: list[BattleState] = []

    async def capture(state: BattleState) -> dict:
        seen.append(state)
        return {}

    use_graph(monkeypatch, finalize=capture)

    await run_battle_graph(trigger=Trigger.BATCH)

    assert seen[0].trigger is Trigger.BATCH


@pytest.mark.asyncio
async def test_discarded_problem_is_none(monkeypatch, caplog) -> None:
    async def always_duplicate(state: BattleState) -> dict:
        return discard(DiscardReason.DUPLICATE, "check_duplicate", "유사도 0.9")

    use_graph(monkeypatch, check_duplicate=always_duplicate)

    with caplog.at_level("INFO"):
        assert await run_battle_graph(trigger=Trigger.BATCH) is None
    assert "DUPLICATE" in caplog.text


@pytest.mark.asyncio
async def test_graph_error_is_none_not_raised(monkeypatch) -> None:
    """배틀 실패가 뒤이은 일반 문제 생성을 멈추면 안 된다."""

    class BrokenGraph:
        async def ainvoke(self, state):
            raise RuntimeError("LLM 연결 끊김")

    monkeypatch.setattr(module, "get_battle_graph", lambda: BrokenGraph())

    assert await run_battle_graph(trigger=Trigger.BATCH) is None
