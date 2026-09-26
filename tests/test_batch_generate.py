import asyncio
import contextlib
import logging
from collections import Counter

import pytest

from src.batch import generate as module
from src.batch.generate import (
    ALL_KEYS,
    MAX_ATTEMPTS,
    TARGET_STOCK,
    fill_slot,
    interleave,
    plan_generation,
    run_generate,
)
from src.core.config import get_settings
from src.core.enums import Category, Difficulty, ProblemPurpose, Trigger
from src.schema.batch import ProblemDemand

DP2 = (Category.DP, Difficulty.LV2)
ARRAY3 = (Category.ARRAY, Difficulty.LV3)
GREEDY1 = (Category.GREEDY, Difficulty.LV1)


def demand(key: tuple[Category, Difficulty], count: int) -> ProblemDemand:
    category, difficulty = key
    return ProblemDemand(difficulty=difficulty, category=category, count=count)


# ── 조합 목록 ──────────────────────────────────────────────────────────


def test_all_keys_cover_every_real_combination() -> None:
    assert len(ALL_KEYS) == (len(Category) - 1) * len(Difficulty)
    assert len(set(ALL_KEYS)) == len(ALL_KEYS)
    assert all(category != Category.RANDOM for category, _ in ALL_KEYS)


# ── 생성 계획 ──────────────────────────────────────────────────────────


def test_empty_stock_fills_every_combination() -> None:
    """첫날 또는 Spring 조회에 실패해 빈 목록을 넘긴 날."""
    plan = plan_generation([], {})

    assert set(plan) == set(ALL_KEYS)
    assert set(plan.values()) == {TARGET_STOCK}


@pytest.mark.parametrize(
    ("spring", "expected"),
    [(0, 3), (1, 2), (2, 1), (3, None), (7, None)],
)
def test_spring_stock_is_topped_up_to_target(spring, expected) -> None:
    plan = plan_generation([demand(DP2, spring)], {})

    assert plan.get(DP2) == expected


def test_combination_missing_from_spring_is_treated_as_empty() -> None:
    plan = plan_generation([demand(DP2, 3)], {})

    assert plan[ARRAY3] == TARGET_STOCK


def test_unsent_buffer_counts_as_stock() -> None:
    """전날 전송에 실패해 남은 문제를 또 만들면 재고가 넘친다."""
    plan = plan_generation([demand(DP2, 1)], {DP2: 1, ARRAY3: 3})

    assert plan[DP2] == 1
    assert ARRAY3 not in plan


def test_need_never_goes_negative() -> None:
    plan = plan_generation([demand(DP2, 5)], {DP2: 5})

    assert DP2 not in plan
    assert all(need > 0 for need in plan.values())


def test_duplicated_spring_entries_are_summed(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        plan = plan_generation([demand(DP2, 1), demand(DP2, 1)], {})

    assert plan[DP2] == 1
    assert "중복" in caplog.text


# ── 순서 섞기 ──────────────────────────────────────────────────────────


def test_every_slot_is_kept() -> None:
    plan = {DP2: 3, ARRAY3: 1, GREEDY1: 2}

    slots = interleave(plan)

    assert Counter(slots) == Counter(plan)


def test_same_combination_is_not_adjacent() -> None:
    slots = interleave({DP2: 3, ARRAY3: 3, GREEDY1: 3})

    assert all(a != b for a, b in zip(slots, slots[1:], strict=False))


def test_first_round_visits_every_combination_once() -> None:
    plan = {DP2: 3, ARRAY3: 1, GREEDY1: 2}

    slots = interleave(plan)

    assert slots[: len(plan)] == list(plan)
    assert slots == [DP2, ARRAY3, GREEDY1, DP2, GREEDY1, DP2]


def test_full_plan_spreads_same_combination_far_apart() -> None:
    slots = interleave(plan_generation([], {}))
    positions = [i for i, key in enumerate(slots) if key == DP2]

    assert len(slots) == len(ALL_KEYS) * TARGET_STOCK
    assert positions[1] - positions[0] == len(ALL_KEYS)


def test_neighbouring_slots_have_different_categories() -> None:
    """중복 검사는 카테고리 단위다. 난이도만 다른 같은 카테고리도 이웃하면 안 된다."""
    categories = len(Category) - 1
    slots = interleave(plan_generation([], {}))

    for start in range(len(slots) - categories + 1):
        window = slots[start : start + categories]
        assert len({category for category, _ in window}) == categories


def test_empty_plan_has_no_slots() -> None:
    assert interleave({}) == []


# ── 실행 ───────────────────────────────────────────────────────────────


class FakeBatch:
    """run_generate가 부르는 바깥 의존성(DB, Spring, 그래프)의 대역."""

    def __init__(self) -> None:
        self.acquired = True
        self.lock_error: Exception | None = None
        # 기본은 모든 조합이 이미 꽉 찬 상태. 테스트가 필요한 조합만 비운다.
        self.demands: list[ProblemDemand] | Exception = [
            demand(key, TARGET_STOCK) for key in ALL_KEYS
        ]
        self.pending: dict[ProblemPurpose, dict] = {
            ProblemPurpose.NORMAL: {},
            ProblemPurpose.DAILY: {},
        }
        self.demand_failures = 0  # 앞에서부터 이만큼은 연결 실패
        self.daily_error: Exception | None = None
        # 조합별로 시도 결과를 순서대로 꺼낸다. 비어 있으면 성공.
        self.outcomes: dict[tuple, list[bool]] = {}

        self.demand_calls = 0
        self.daily_calls: list[Trigger] = []
        self.graph_calls: list[tuple] = []
        self.running = 0
        self.peak = 0

    def set_spring(self, key: tuple, count: int) -> None:
        assert isinstance(self.demands, list)
        self.demands = [d for d in self.demands if (d.category, d.difficulty) != key]
        self.demands.append(demand(key, count))

    @contextlib.asynccontextmanager
    async def lock(self, key):
        if self.lock_error:
            raise self.lock_error
        yield self.acquired

    async def get_problem_demands(self):
        self.demand_calls += 1
        if self.demand_calls <= self.demand_failures:
            raise ConnectionError("잠깐 끊김")
        if isinstance(self.demands, Exception):
            raise self.demands
        return self.demands

    async def load_pending(self, purpose):
        return self.pending[purpose]

    async def generate_daily_problems(self, trigger):
        self.daily_calls.append(trigger)
        if self.daily_error:
            raise self.daily_error
        return [object()] * 5

    async def run_problem_graph(self, difficulty, category, *, trigger, purpose):
        self.graph_calls.append((category, difficulty, trigger, purpose))
        self.running += 1
        self.peak = max(self.peak, self.running)
        await asyncio.sleep(0)  # 다른 슬롯이 끼어들 틈
        self.running -= 1
        outcomes = self.outcomes.get((category, difficulty))
        succeeded = outcomes.pop(0) if outcomes else True
        return object() if succeeded else None

    def keys_called(self) -> list[tuple]:
        return [
            (category, difficulty) for category, difficulty, _, _ in self.graph_calls
        ]


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeBatch:
    batch = FakeBatch()
    monkeypatch.setattr(module, "try_advisory_lock", batch.lock)
    monkeypatch.setattr(module, "get_problem_demands", batch.get_problem_demands)
    monkeypatch.setattr(module, "load_pending", batch.load_pending)
    monkeypatch.setattr(
        module, "generate_daily_problems", batch.generate_daily_problems
    )
    monkeypatch.setattr(module, "run_problem_graph", batch.run_problem_graph)
    monkeypatch.setattr(module, "DEMAND_RETRY_DELAYS_S", (0, 0))
    return batch


@pytest.mark.asyncio
async def test_another_instance_holding_the_lock_skips_everything(fake) -> None:
    fake.acquired = False
    fake.set_spring(DP2, 0)

    await run_generate()

    assert fake.daily_calls == []
    assert fake.demand_calls == 0
    assert fake.graph_calls == []


@pytest.mark.asyncio
async def test_only_the_shortfall_is_made_as_batch_normal(fake) -> None:
    fake.set_spring(DP2, 0)
    fake.set_spring(ARRAY3, 2)
    fake.pending[ProblemPurpose.NORMAL] = {DP2: 1}

    await run_generate()

    assert Counter(fake.keys_called()) == Counter({DP2: 2, ARRAY3: 1})
    assert {(t, p) for _, _, t, p in fake.graph_calls} == {
        (Trigger.BATCH, ProblemPurpose.NORMAL)
    }


@pytest.mark.asyncio
async def test_spring_failure_falls_back_to_buffer_stock(fake, caplog) -> None:
    fake.demands = ConnectionError("Spring 배포 중")
    fake.pending[ProblemPurpose.NORMAL] = {DP2: 3}

    with caplog.at_level(logging.WARNING):
        await run_generate()

    assert fake.demand_calls == 1 + len(module.DEMAND_RETRY_DELAYS_S)
    assert len(fake.graph_calls) == len(ALL_KEYS) * TARGET_STOCK - 3
    assert DP2 not in fake.keys_called()
    assert "버퍼 재고만 보고" in caplog.text


@pytest.mark.asyncio
async def test_spring_recovering_on_retry_is_used(fake) -> None:
    fake.demand_failures = 1  # 첫 시도만 실패
    fake.set_spring(DP2, 2)

    await run_generate()

    assert fake.demand_calls == 2
    assert fake.keys_called() == [DP2]  # fallback이었다면 240개 가까이 만든다


@pytest.mark.asyncio
async def test_spring_failure_does_not_stop_daily(fake) -> None:
    fake.demands = ConnectionError("Spring 배포 중")

    await run_generate()

    assert fake.daily_calls == [Trigger.BATCH]


@pytest.mark.asyncio
async def test_daily_failure_does_not_stop_normal(fake, caplog) -> None:
    fake.daily_error = RuntimeError("데일리 실패")
    fake.set_spring(DP2, 2)

    with caplog.at_level(logging.ERROR):
        await run_generate()

    assert fake.keys_called() == [DP2]
    assert "generate_daily" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(("pending", "made"), [(0, True), (4, True), (5, False)])
async def test_daily_is_made_only_when_pending_is_short(fake, pending, made) -> None:
    fake.pending[ProblemPurpose.DAILY] = {DP2: pending}

    await run_generate()

    assert fake.daily_calls == ([Trigger.BATCH] if made else [])


@pytest.mark.asyncio
async def test_concurrency_never_exceeds_the_limit(fake, monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "batch_concurrency", 3)
    fake.demands = []  # 240개 전부

    await run_generate()

    assert len(fake.graph_calls) == len(ALL_KEYS) * TARGET_STOCK
    assert fake.peak == 3


@pytest.mark.asyncio
async def test_missed_combinations_are_logged(fake, caplog) -> None:
    fake.set_spring(DP2, 2)
    fake.outcomes[DP2] = [False] * MAX_ATTEMPTS

    with caplog.at_level(logging.WARNING):
        await run_generate()

    assert "채우지 못한 조합: DP/LV2×1" in caplog.text


@pytest.mark.asyncio
async def test_lock_error_is_logged_not_raised(fake, caplog) -> None:
    """스케줄러까지 예외가 올라가면 원인 없이 잡 실패로만 남는다."""
    fake.lock_error = OSError("DB 연결 거부")

    with caplog.at_level(logging.ERROR):
        await run_generate()

    assert "생성 배치를 시작하지 못함" in caplog.text
    assert fake.daily_calls == []


# ── 슬롯 ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_slot_stops_at_first_success(fake) -> None:
    fake.outcomes[DP2] = [False, True]

    assert await fill_slot(DP2, asyncio.Semaphore(1)) is True
    assert len(fake.graph_calls) == 2


@pytest.mark.asyncio
async def test_slot_gives_up_after_max_attempts(fake) -> None:
    fake.outcomes[DP2] = [False] * (MAX_ATTEMPTS + 1)

    assert await fill_slot(DP2, asyncio.Semaphore(1)) is False
    assert len(fake.graph_calls) == MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_failed_slot_retries_behind_waiting_slots(fake) -> None:
    """재시도가 바로 붙어 돌면 같은 조합이 연달아 만들어진다."""
    fake.outcomes[DP2] = [False, True]
    semaphore = asyncio.Semaphore(1)

    await asyncio.gather(fill_slot(DP2, semaphore), fill_slot(ARRAY3, semaphore))

    assert fake.keys_called() == [DP2, ARRAY3, DP2]
