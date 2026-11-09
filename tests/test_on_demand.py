import uuid

import httpx
import pytest

from src.core.enums import Category, Difficulty
from src.problem import on_demand
from src.router import problem as router
from src.schema.problem import Problem, ProblemRequest

PROBLEM_ID = uuid.uuid4()
REQUEST = ProblemRequest.model_validate({"difficulty": "LV1", "category": "DP"})


def make_problem() -> Problem:
    return Problem(
        problem_title="합이 M인 쌍의 개수",
        problem_content="합이 M인 쌍의 수를 구하라",
        input_format="N M",
        output_format="쌍의 수",
        requested_difficulty=Difficulty.LV2,
        difficulty=Difficulty.LV2,
        category=Category.HASH,
        category_select_reason="해시맵으로 푼다",
        input_constraints=[],
        execution_limits=[],
        problem_examples=[],
    )


class FakeRequest:
    """연결 상태만 흉내 내는 요청."""

    def __init__(self, disconnected: bool) -> None:
        self.disconnected = disconnected

    async def is_disconnected(self) -> bool:
        return self.disconnected


class FakeGraph:
    """검증을 통과해 저장까지 끝난 것처럼 결과를 돌려준다."""

    async def ainvoke(self, state):
        return {**state.model_dump(), "problem_id": PROBLEM_ID}


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """생성은 성공한 것으로 두고, 어떤 전달 처리가 불렸는지 기록한다."""
    recorded: list[str] = []

    async def healthy() -> bool:
        return True

    async def save_undelivered(problem_id, problem) -> None:
        recorded.append("save_undelivered")

    async def mark_delivered(problem_id) -> None:
        recorded.append("mark_delivered")

    monkeypatch.setattr(router, "is_healthy", healthy)
    monkeypatch.setattr(router, "get_graph", lambda: FakeGraph())
    monkeypatch.setattr(router, "build_problem", lambda result: make_problem())
    monkeypatch.setattr(router, "save_undelivered", save_undelivered)
    monkeypatch.setattr(router, "mark_delivered", mark_delivered)
    return recorded


@pytest.mark.asyncio
async def test_연결이_끊겼으면_저장_API로_보낸다(calls):
    await router.make_problem(REQUEST, FakeRequest(disconnected=True))

    assert calls == ["save_undelivered"]


@pytest.mark.asyncio
async def test_연결돼_있으면_응답하고_보냄_표시만_한다(calls):
    response = await router.make_problem(REQUEST, FakeRequest(disconnected=False))

    assert calls == ["mark_delivered"]
    assert response.problem.problem_title == "합이 M인 쌍의 개수"


@pytest.mark.asyncio
async def test_저장에_성공하면_보냄_표시를_한다(monkeypatch):
    sent: list[Problem] = []
    marked: list[uuid.UUID] = []

    async def save_problems(problems: list[Problem]) -> None:
        sent.extend(problems)

    async def mark_delivered(problem_id: uuid.UUID) -> None:
        marked.append(problem_id)

    monkeypatch.setattr(on_demand, "save_problems", save_problems)
    monkeypatch.setattr(on_demand, "mark_delivered", mark_delivered)

    await on_demand.save_undelivered(PROBLEM_ID, make_problem())

    assert len(sent) == 1
    assert marked == [PROBLEM_ID]


@pytest.mark.asyncio
async def test_저장에_실패하면_표시하지_않고_넘어간다(monkeypatch):
    marked: list[uuid.UUID] = []

    async def save_problems(problems: list[Problem]) -> None:
        raise httpx.ConnectError("연결 실패")

    async def mark_delivered(problem_id: uuid.UUID) -> None:
        marked.append(problem_id)

    monkeypatch.setattr(on_demand, "save_problems", save_problems)
    monkeypatch.setattr(on_demand, "mark_delivered", mark_delivered)

    await on_demand.save_undelivered(PROBLEM_ID, make_problem())

    assert marked == []
