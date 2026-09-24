import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exception import ProblemGenerationError
from src.core.exception_handler import register_exception_handlers
from src.core.logging import setup_logging
from src.db.session import close_engine, get_session
from src.problem.graph import build_graph
from src.problem.nodes.finalize import build_problem
from src.problem.state import GraphState
from src.schema.evaluation import EvaluationRequest, EvaluationResponse
from src.schema.problem import (
    BattleProblemResponse,
    DailyProblemResponse,
    ProblemRequest,
    ProblemResponse,
)

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


@lru_cache(maxsize=1)
def get_graph():
    """그래프는 한 번만 조립한다. 요청마다 다시 만들 이유가 없다."""
    return build_graph()


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
async def make_problem(request: ProblemRequest) -> ProblemResponse:
    """문제 하나를 생성한다. 검증에서 걸리면 폐기되고 502로 응답한다."""
    state = GraphState(
        requested_difficulty=request.difficulty,
        requested_category=request.category,
    )
    result = GraphState(**await get_graph().ainvoke(state))

    if result.is_discarded:
        # 자세한 사유는 discard_problem이 로그와 DB에 남긴다
        raise ProblemGenerationError(
            f"문제를 만들지 못했습니다 "
            f"({result.discard_stage}: {result.discard_reason})"
        )

    return ProblemResponse(problem=build_problem(result))


@app.post("/api/llm/problem/daily")
async def makeDailyProblem() -> DailyProblemResponse:
    return {"status": "ok"}


@app.post("/api/llm/problem/battle")
async def makeBattleProblem() -> BattleProblemResponse:
    return {"status": "ok"}


@app.post("/api/llm/evaluation")
async def evaluate_nl_solution(request: EvaluationRequest) -> EvaluationResponse:
    return EvaluationResponse()
