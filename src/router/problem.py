from fastapi import APIRouter

from src.core.exception import ProblemGenerationError
from src.problem.battle.graph import get_battle_graph
from src.problem.battle.state import BattleState
from src.problem.daily import generate_daily_problems
from src.problem.graph import get_graph
from src.problem.nodes.finalize import build_problem
from src.problem.state import GraphState
from src.schema.problem import (
    BattleProblem,
    BattleProblemResponse,
    DailyProblemResponse,
    HiddenTestCase,
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
    """데일리 문제 5개를 생성한다. 하나도 만들지 못하면 502로 응답한다."""
    problems = await generate_daily_problems()

    if not problems:
        raise ProblemGenerationError("데일리 문제를 하나도 만들지 못했습니다")

    return DailyProblemResponse(problems=problems)


@router.post("/api/llm/problem/battle")
async def make_battle_problem() -> BattleProblemResponse:
    """배틀 문제 하나를 생성한다. 검증에서 걸리면 폐기되고 502로 응답한다."""
    result = BattleState(**await get_battle_graph().ainvoke(BattleState()))

    if result.is_discarded:
        raise ProblemGenerationError(
            f"배틀 문제를 만들지 못했습니다 "
            f"({result.discard_stage}: {result.discard_reason})"
        )

    return BattleProblemResponse(
        battle_problem=BattleProblem(
            category=result.category,
            problem_title=result.problem_title,
            problem_content=result.problem_content,
            test_cases=[
                HiddenTestCase(input=case.input, output=case.output)
                for case in result.test_cases
            ],
        )
    )
