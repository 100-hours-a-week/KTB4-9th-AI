from fastapi import APIRouter

from src.core.exception import ProblemGenerationError
from src.problem.graph import get_graph
from src.problem.nodes.finalize import build_problem
from src.problem.state import GraphState
from src.schema.problem import (
    BattleProblemResponse,
    DailyProblemResponse,
    ProblemRequest,
    ProblemResponse,
)

router = APIRouter()


@router.post("/api/llm/problem")
async def make_problem(request: ProblemRequest) -> ProblemResponse:
    """문제 하나를 생성한다. 검증에서 걸리면 폐기되고 502로 응답한다."""
    state = GraphState(
        requested_difficulty=request.difficulty,
        requested_category=request.category,
    )
    result = GraphState(**await get_graph().ainvoke(state))

    if result.is_discarded:
        raise ProblemGenerationError(
            f"문제를 만들지 못했습니다 "
            f"({result.discard_stage}: {result.discard_reason})"
        )

    return ProblemResponse(problem=build_problem(result))


@router.post("/api/llm/problem/daily")
async def make_daily_problem() -> DailyProblemResponse:
    return {"status": "ok"}


@router.post("/api/llm/problem/battle")
async def make_battle_problem() -> BattleProblemResponse:
    return {"status": "ok"}
