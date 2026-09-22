from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.schema import (
    BattleProblemResponse,
    DailyProblemResponse,
    EvaluationRequest,
    EvaluationResponse,
    ProblemRequest,
    ProblemResponse,
)
from src.shared.db_client import get_session

app = FastAPI()


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
