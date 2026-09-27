from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.core.config import get_settings

_settings = get_settings()

engine = create_async_engine(
    _settings.db_url,
    echo=_settings.db_echo,
    pool_size=_settings.db_pool_size,
    max_overflow=_settings.db_max_overflow,
    pool_pre_ping=True,  # 커넥션 끊김 감지 (RDS idle timeout 대비)
)

SessionFactory = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession]:
    """FastAPI 의존성. 요청 1건 = 트랜잭션 1건, 예외 시 자동 롤백."""
    async with SessionFactory() as session, session.begin():
        yield session


@asynccontextmanager
async def session_scope() -> AsyncGenerator[AsyncSession]:
    """배치 러너·LangGraph 노드처럼 FastAPI 밖에서 쓰는 경로용."""
    async with SessionFactory() as session, session.begin():
        yield session


async def close_engine() -> None:
    """FastAPI lifespan 종료 시, 배치 프로세스 종료 시 호출."""
    await engine.dispose()
