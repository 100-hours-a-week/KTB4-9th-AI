import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.handler import register_exception_handlers
from src.core.logging import setup_logging
from src.schema import (
    BattleProblemResponse,
    DailyProblemResponse,
    EvaluationRequest,
    EvaluationResponse,
    ProblemRequest,
    ProblemResponse,
)
from src.shared.db_client import close_engine, get_session

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


@app.get("/db-check")
async def checkDB(session: Annotated[AsyncSession, Depends(get_session)]):
    try:
        await session.execute(text("SELECT 1"))
    except SQLAlchemyError:
        raise HTTPException(503, "database unavailable") from None
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/llm/problem")
async def makeProble(request: ProblemRequest) -> ProblemResponse:
    return {"status": "ok"}


@app.post("/api/llm/problem/daily")
async def makeDailyProblem() -> DailyProblemResponse:
    return {"status": "ok"}


@app.post("/api/llm/problem/battle")
async def makeBattleProblem() -> BattleProblemResponse:
    return {"status": "ok"}


@app.post("/api/llm/evaluation")
async def evaluate_nl_solution(request: EvaluationRequest) -> EvaluationResponse:
    return EvaluationResponse()
