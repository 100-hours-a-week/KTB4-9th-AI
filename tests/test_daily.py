import pytest

from src.core.enums import Category, Difficulty, ProblemPurpose, Trigger
from src.problem import daily as module
from src.problem.daily import (
    DAILY_CATEGORIES,
    DAILY_COUNT,
    DAILY_DIFFICULTIES,
    generate_daily_problems,
)


class FakeProblem:
    """Problem 대신 개수만 세기 위한 가짜 객체."""


def fix_generate_one(
    monkeypatch: pytest.MonkeyPatch, results: list[list[object | None]]
) -> list[int]:
    """
    회차마다 정해진 결과를 돌려주도록 generate_one을 바꿔 끼운다.

    Parameters:
        results (list[list[object | None]]): 회차별 반환값 목록

    Returns:
        list[int]: 회차마다 호출된 횟수가 쌓이는 목록
    """
    calls: list[int] = []
    rounds = iter(results)
    current: list[object | None] = []

    async def fake(category, trigger) -> object | None:
        if not current:
            batch = next(rounds)
            calls.append(len(batch))
            current.extend(batch)
        return current.pop(0)

    monkeypatch.setattr(module, "generate_one", fake)
    return calls


@pytest.mark.asyncio
async def test_한_번에_다_성공하면_재시도하지_않는다(monkeypatch):
    calls = fix_generate_one(monkeypatch, [[FakeProblem()] * 5])
    problems = await generate_daily_problems(Trigger.ON_DEMAND)

    assert len(problems) == DAILY_COUNT
    assert calls == [5]


@pytest.mark.asyncio
async def test_부족하면_부족한_만큼만_다시_시도한다(monkeypatch):
    calls = fix_generate_one(
        monkeypatch,
        [
            [FakeProblem(), FakeProblem(), FakeProblem(), None, None],
            [FakeProblem(), FakeProblem()],
        ],
    )
    problems = await generate_daily_problems(Trigger.ON_DEMAND)

    assert len(problems) == DAILY_COUNT
    assert calls == [5, 2]


@pytest.mark.asyncio
async def test_세_번_시도해도_부족하면_나온_만큼_반환한다(monkeypatch):
    calls = fix_generate_one(
        monkeypatch,
        [
            [FakeProblem()] + [None] * 4,
            [FakeProblem()] + [None] * 3,
            [FakeProblem()] + [None] * 2,
        ],
    )
    problems = await generate_daily_problems(Trigger.ON_DEMAND)

    assert len(problems) == 3
    assert calls == [5, 4, 3]


@pytest.mark.asyncio
async def test_하나도_못_만들면_빈_목록을_반환한다(monkeypatch):
    fix_generate_one(monkeypatch, [[None] * 5, [None] * 5, [None] * 5])
    problems = await generate_daily_problems(Trigger.ON_DEMAND)

    assert problems == []


def test_난이도는_LV1과_LV2만_쓴다():
    assert DAILY_DIFFICULTIES == [Difficulty.LV1, Difficulty.LV2]


def test_카테고리에_RANDOM은_없다():
    assert Category.RANDOM not in DAILY_CATEGORIES
    assert len(DAILY_CATEGORIES) > 1


@pytest.mark.asyncio
async def test_한_회차_안에서_카테고리가_겹치지_않는다(monkeypatch):
    seen: list[object] = []

    async def fake(category, trigger):
        seen.append(category)
        return FakeProblem()

    monkeypatch.setattr(module, "generate_one", fake)
    await generate_daily_problems(Trigger.ON_DEMAND)

    assert len(seen) == DAILY_COUNT
    assert len(set(seen)) == DAILY_COUNT


@pytest.mark.asyncio
async def test_생성_경로가_문제마다_그대로_전달된다(monkeypatch):
    triggers: list[Trigger] = []

    async def fake(category, trigger):
        triggers.append(trigger)
        return FakeProblem()

    monkeypatch.setattr(module, "generate_one", fake)
    await generate_daily_problems(Trigger.BATCH)

    assert triggers == [Trigger.BATCH] * DAILY_COUNT


@pytest.mark.asyncio
async def test_데일리_문제는_DAILY_용도로_쉬운_난이도에서_만든다(monkeypatch):
    """purpose가 NORMAL로 저장되면 데일리가 /problems로 나간다."""
    received: list[tuple] = []

    async def fake_run(difficulty, category, *, trigger, purpose):
        received.append((difficulty, category, trigger, purpose))
        return None

    monkeypatch.setattr(module, "run_problem_graph", fake_run)

    await module.generate_one(Category.DP, Trigger.BATCH)

    difficulty, category, trigger, purpose = received[0]
    assert difficulty in DAILY_DIFFICULTIES
    assert category is Category.DP
    assert trigger is Trigger.BATCH
    assert purpose is ProblemPurpose.DAILY
