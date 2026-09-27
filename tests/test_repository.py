from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.enums import Category, Difficulty, ProblemPurpose, Trigger
from src.db.models import BattleProblem
from src.db.repository import BattleProblemRepository, GeneratedProblemRepository
from src.db.session import engine
from src.schema.problem import Problem

KEY = (Category.HASH_TABLE, Difficulty.LV2)


@pytest_asyncio.fixture
async def session(db_available: None) -> AsyncGenerator[AsyncSession]:
    """테스트가 끝나면 통째로 롤백한다. 로컬 DB에 흔적을 남기지 않는다."""
    async with engine.connect() as conn:
        outer = await conn.begin()
        session = AsyncSession(bind=conn, join_transaction_mode="create_savepoint")
        try:
            yield session
        finally:
            await session.close()
            await outer.rollback()


@pytest.fixture
def repo(session: AsyncSession) -> GeneratedProblemRepository:
    return GeneratedProblemRepository(session)


@pytest.fixture
def battle_repo(session: AsyncSession) -> BattleProblemRepository:
    return BattleProblemRepository(session)


def make_problem(
    category: Category = KEY[0], difficulty: Difficulty = KEY[1]
) -> Problem:
    return Problem(
        problem_title="두 수의 합",
        problem_content="합이 M인 쌍의 수를 구하라",
        input_format="N M",
        output_format="쌍의 수",
        requested_difficulty=difficulty,
        difficulty=difficulty,
        category=category,
        category_select_reason="해시맵으로 푼다",
        input_constraints=[],
        execution_limits=[],
        problem_examples=[],
    )


# ── 저장 ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_stores_given_trigger_and_purpose(
    repo: GeneratedProblemRepository,
) -> None:
    row = await repo.add(
        make_problem(), trigger=Trigger.BATCH, purpose=ProblemPurpose.DAILY
    )
    await repo.session.refresh(row)  # DB에 실제로 들어간 값으로 확인

    assert row.trigger is Trigger.BATCH
    assert row.purpose is ProblemPurpose.DAILY


# ── 미전송 재고 ────────────────────────────────────────────────────────
# 로컬 DB에 이미 있는 행과 섞이므로 넣기 전후 차이로 본다.


@pytest.mark.asyncio
async def test_count_pending_counts_only_batch_rows_of_that_purpose(
    repo: GeneratedProblemRepository,
) -> None:
    before = await repo.count_pending()

    await repo.add(make_problem(), trigger=Trigger.BATCH, purpose=ProblemPurpose.NORMAL)
    await repo.add(make_problem(), trigger=Trigger.BATCH, purpose=ProblemPurpose.NORMAL)
    await repo.add(
        make_problem(), trigger=Trigger.ON_DEMAND, purpose=ProblemPurpose.NORMAL
    )
    await repo.add(make_problem(), trigger=Trigger.BATCH, purpose=ProblemPurpose.DAILY)
    await repo.add(
        make_problem(Category.DP, Difficulty.LV1),
        trigger=Trigger.BATCH,
        purpose=ProblemPurpose.NORMAL,
    )

    after = await repo.count_pending()

    assert after[KEY] - before.get(KEY, 0) == 2
    dp_key = (Category.DP, Difficulty.LV1)
    assert after[dp_key] - before.get(dp_key, 0) == 1


@pytest.mark.asyncio
async def test_count_pending_keys_are_enums(repo: GeneratedProblemRepository) -> None:
    await repo.add(make_problem(), trigger=Trigger.BATCH, purpose=ProblemPurpose.NORMAL)

    category, difficulty = next(iter(await repo.count_pending()))

    assert isinstance(category, Category)
    assert isinstance(difficulty, Difficulty)


@pytest.mark.asyncio
async def test_list_pending_returns_only_batch_rows_of_that_purpose(
    repo: GeneratedProblemRepository,
) -> None:
    normal = await repo.add(
        make_problem(), trigger=Trigger.BATCH, purpose=ProblemPurpose.NORMAL
    )
    daily = await repo.add(
        make_problem(), trigger=Trigger.BATCH, purpose=ProblemPurpose.DAILY
    )
    on_demand = await repo.add(
        make_problem(), trigger=Trigger.ON_DEMAND, purpose=ProblemPurpose.NORMAL
    )

    normal_ids = {r.id for r in await repo.list_pending(ProblemPurpose.NORMAL, 1000)}
    daily_ids = {r.id for r in await repo.list_pending(ProblemPurpose.DAILY, 1000)}

    assert normal.id in normal_ids
    assert daily.id in daily_ids
    assert daily.id not in normal_ids
    assert on_demand.id not in normal_ids | daily_ids


@pytest.mark.asyncio
async def test_sent_rows_leave_the_pending_stock(
    repo: GeneratedProblemRepository,
) -> None:
    before = await repo.count_pending()
    row = await repo.add(
        make_problem(), trigger=Trigger.BATCH, purpose=ProblemPurpose.NORMAL
    )

    assert await repo.mark_sent([row.id]) == 1

    after = await repo.count_pending()
    assert after.get(KEY, 0) == before.get(KEY, 0)
    pending_ids = {r.id for r in await repo.list_pending(ProblemPurpose.NORMAL, 1000)}
    assert row.id not in pending_ids


@pytest.mark.asyncio
async def test_list_pending_skips_excluded_rows(
    repo: GeneratedProblemRepository,
) -> None:
    """거부된 행이 매번 첫 청크를 차지하면 나머지가 영영 못 나간다."""
    rejected = await repo.add(
        make_problem(), trigger=Trigger.BATCH, purpose=ProblemPurpose.NORMAL
    )
    fine = await repo.add(
        make_problem(), trigger=Trigger.BATCH, purpose=ProblemPurpose.NORMAL
    )

    rows = await repo.list_pending(
        ProblemPurpose.NORMAL, 1000, exclude_ids={rejected.id}
    )
    ids = {r.id for r in rows}

    assert rejected.id not in ids
    assert fine.id in ids


# ── 배틀 ───────────────────────────────────────────────────────────────
# 로컬 DB에 이미 있는 행과 섞이므로 넣기 전후 차이로 본다.


async def add_battle(
    repo: BattleProblemRepository, trigger: Trigger = Trigger.BATCH
) -> BattleProblem:
    return await repo.add(
        category=Category.ARRAY,
        problem_title="배열 합",
        problem_content="N개의 정수 합을 출력하라",
        test_cases=[{"input": "1\n5", "output": "5"}],
        trigger=trigger,
    )


@pytest.mark.asyncio
async def test_battle_add_stores_given_trigger(
    battle_repo: BattleProblemRepository,
) -> None:
    row = await add_battle(battle_repo, Trigger.BATCH)
    await battle_repo.session.refresh(row)  # DB에 실제로 들어간 값으로 확인

    assert row.trigger is Trigger.BATCH
    assert row.sent_at is None


@pytest.mark.asyncio
async def test_battle_count_pending_counts_only_unsent_batch_rows(
    battle_repo: BattleProblemRepository,
) -> None:
    before = await battle_repo.count_pending()

    await add_battle(battle_repo, Trigger.BATCH)
    sent = await add_battle(battle_repo, Trigger.BATCH)
    await add_battle(battle_repo, Trigger.ON_DEMAND)
    await battle_repo.mark_sent([sent.id])

    assert await battle_repo.count_pending() - before == 1


@pytest.mark.asyncio
async def test_battle_list_pending_skips_on_demand_and_sent_rows(
    battle_repo: BattleProblemRepository,
) -> None:
    pending = await add_battle(battle_repo, Trigger.BATCH)
    sent = await add_battle(battle_repo, Trigger.BATCH)
    on_demand = await add_battle(battle_repo, Trigger.ON_DEMAND)
    await battle_repo.mark_sent([sent.id])

    ids = {row.id for row in await battle_repo.list_pending(limit=1000)}

    assert pending.id in ids
    assert sent.id not in ids
    assert on_demand.id not in ids


@pytest.mark.asyncio
async def test_battle_list_pending_starts_from_the_oldest(
    battle_repo: BattleProblemRepository,
) -> None:
    """못 보낸 문제가 먼저 나가야 그날 만든 문제가 다음 날로 밀린다."""
    await add_battle(battle_repo, Trigger.BATCH)
    oldest = await add_battle(battle_repo, Trigger.BATCH)
    # 같은 트랜잭션 안에서는 now()가 같으므로 시각을 직접 앞당긴다.
    oldest.created_at = datetime(2000, 1, 1, tzinfo=UTC)
    await battle_repo.session.flush()

    (row,) = await battle_repo.list_pending(limit=1)

    assert row.id == oldest.id


@pytest.mark.asyncio
async def test_battle_mark_sent_only_changes_unsent_rows(
    battle_repo: BattleProblemRepository,
) -> None:
    row = await add_battle(battle_repo, Trigger.BATCH)

    assert await battle_repo.mark_sent([row.id]) == 1
    assert await battle_repo.mark_sent([row.id]) == 0
    assert await battle_repo.mark_sent([]) == 0
