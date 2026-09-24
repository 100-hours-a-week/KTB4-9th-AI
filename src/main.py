import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.core.exception_handler import register_exception_handlers
from src.core.logging import setup_logging
from src.db.session import close_engine
from src.router.evaluation import router as evaluation_router
from src.router.health import router as health_router
from src.router.problem import router as problem_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    setup_logging()
    logger.info("애플리케이션 시작")
    yield
    await close_engine()  # 커넥션 풀 정리
    logger.info("애플리케이션 종료")


app = FastAPI(lifespan=lifespan)

register_exception_handlers(app)

app.include_router(health_router)
app.include_router(problem_router)
app.include_router(evaluation_router)
