import pytest
from sqlalchemy import text

from src.core.enums import LockKey
from src.db.lock import try_advisory_lock
from src.db.session import engine

pytestmark = pytest.mark.usefixtures("db_available")


async def held_count(key: LockKey) -> int:
    """서버에 남아 있는 해당 키의 advisory lock 수. 풀 커넥션에 샌 것까지 보인다."""
    async with engine.connect() as conn:
        return await conn.scalar(
            text(
                "SELECT count(*) FROM pg_locks "
                "WHERE locktype = 'advisory' AND objid = :key AND granted"
            ),
            {"key": int(key)},
        )


@pytest.mark.asyncio
async def test_second_attempt_is_skipped_while_held() -> None:
    async with try_advisory_lock(LockKey.BATCH_GENERATE) as first:
        async with try_advisory_lock(LockKey.BATCH_GENERATE) as second:
            assert first is True
            assert second is False


@pytest.mark.asyncio
async def test_different_keys_do_not_block_each_other() -> None:
    async with try_advisory_lock(LockKey.BATCH_GENERATE) as generate:
        async with try_advisory_lock(LockKey.BATCH_SEND) as send:
            assert generate is True
            assert send is True


@pytest.mark.asyncio
async def test_lock_is_released_after_the_block() -> None:
    async with try_advisory_lock(LockKey.BATCH_GENERATE) as acquired:
        assert acquired is True
        assert await held_count(LockKey.BATCH_GENERATE) == 1

    assert await held_count(LockKey.BATCH_GENERATE) == 0


@pytest.mark.asyncio
async def test_lock_is_released_when_the_job_fails() -> None:
    with pytest.raises(RuntimeError):
        async with try_advisory_lock(LockKey.BATCH_GENERATE):
            raise RuntimeError("작업 실패")

    assert await held_count(LockKey.BATCH_GENERATE) == 0


@pytest.mark.asyncio
async def test_skipped_attempt_does_not_release_the_holder() -> None:
    async with try_advisory_lock(LockKey.BATCH_GENERATE):
        async with try_advisory_lock(LockKey.BATCH_GENERATE) as second:
            assert second is False
        assert await held_count(LockKey.BATCH_GENERATE) == 1
