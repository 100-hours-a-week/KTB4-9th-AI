from collections.abc import AsyncGenerator

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from src.client import judge0
from src.db.session import engine


@pytest.fixture(autouse=True)
def disable_langfuse_in_tests(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """유닛 테스트 시 실제 Langfuse 서버로 트레이스가 전송되지 않게 차단한다."""
    if request.node.fspath.basename == "test_langfuse.py":
        return
    monkeypatch.setattr("src.client.langfuse.is_langfuse_enabled", lambda: False)
    monkeypatch.setattr("src.client.langfuse.get_langfuse_client", lambda: None)
    monkeypatch.setattr(
        "src.client.langfuse.get_langfuse_callback_handler", lambda **_: None
    )


@pytest.fixture
def judge0_available() -> None:
    """judge0에 실제로 제출하는 테스트용. judge0가 떠 있지 않으면 건너뛴다."""
    try:
        httpx.get(f"{judge0.JUDGE0_URL}/about", timeout=1.0).raise_for_status()
    except httpx.HTTPError as error:
        pytest.skip(f"judge0에 연결할 수 없음 ({judge0.JUDGE0_URL}): {error!r}")


@pytest_asyncio.fixture
async def db_available() -> AsyncGenerator[None]:
    """실제 Postgres가 필요한 테스트용. DB가 떠 있지 않으면 건너뛴다.

    엔진 풀의 커넥션은 이벤트 루프에 묶이는데, 테스트마다 루프가 새로 생긴다.
    다음 테스트가 죽은 루프의 커넥션을 받지 않도록 끝날 때 풀을 비운다.
    """
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except OperationalError as error:
        await engine.dispose()
        pytest.skip(f"DB에 연결할 수 없음: {error.orig!r}")
    yield
    await engine.dispose()
