"""배치 중복 실행 방지.

blue/green 전환 중에는 두 컨테이너가 같은 시각에 같은 잡을 띄울 수 있다.
Postgres advisory lock을 먼저 잡은 쪽만 실행하고 나머지는 건너뛴다.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from src.core.enums import LockKey
from src.db.session import engine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def try_advisory_lock(key: LockKey) -> AsyncGenerator[bool]:
    """
    lock을 시도하고 잡았는지를 넘긴다. 기다리지 않는다.

    세션 단위 잠금이라 잡은 커넥션에서만 풀 수 있고, 커넥션이 살아 있는 동안
    유지된다. 그래서 작업 내내 전용 커넥션을 쥐고 있다가 같은 커넥션에서 푼다.
    풀에 반납된 커넥션에 잠금이 남으면 다음 날 배치가 영영 건너뛰게 된다.

    AUTOCOMMIT으로 여는 이유: 몇 시간짜리 작업 동안 트랜잭션을 열어 두면
    idle in transaction으로 남아 vacuum을 막는다.

    Parameters:
        key (LockKey): 잠금 키

    Yields:
        bool: lock을 잡았으면 True. False면 다른 인스턴스가 실행 중이다
    """
    async with engine.connect() as conn:
        conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
        acquired = await conn.scalar(
            text("SELECT pg_try_advisory_lock(:key)"), {"key": int(key)}
        )
        try:
            yield acquired
        finally:
            if acquired:
                await _release(conn, key)


async def _release(conn: AsyncConnection, key: LockKey) -> None:
    """lock을 푼다. 풀지 못하면 커넥션을 버려 lock이 풀로 새지 않게 한다."""
    try:
        await conn.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": int(key)})
    except Exception:
        # 커넥션이 끊겼다면 서버 쪽 lock은 이미 풀렸다.
        logger.warning("advisory lock 해제 실패 (key=%s)", key.name, exc_info=True)
        await conn.invalidate()
