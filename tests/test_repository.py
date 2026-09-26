from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.enums import Category, Difficulty, ProblemPurpose, Trigger
from src.db.repository import GeneratedProblemRepository
from src.db.session import engine
from src.schema.problem import Problem

KEY = (Category.HASH_TABLE, Difficulty.LV2)


@pytest_asyncio.fixture
async def repo(db_available: None) -> AsyncGenerator[GeneratedProblemRepository]:
    """테스트가 끝나면 통째로 롤백한다. 로컬 DB에 흔적을 남기지 않는다."""
    async with engine.connect() as conn:
        outer = await conn.begin()
        session = AsyncSession(bind=conn, join_transaction_mode="create_savepoint")
        try:
            yield GeneratedProblemRepository(session)
        finally:
            await session.close()
            await outer.rollback()


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
