from fastapi import APIRouter, Request

from src.client.judge0 import is_healthy
from src.core.enums import Trigger
from src.core.exception import JudgeUnavailableError, ProblemGenerationError
from src.problem.battle.graph import get_battle_graph
from src.problem.battle.state import BattleState
from src.problem.daily import generate_daily_problems
from src.problem.graph import get_graph
from src.problem.nodes.finalize import build_problem
from src.problem.on_demand import mark_delivered, save_undelivered
from src.problem.state import GraphState
from src.schema.problem import (
    BattleProblemResponse,
    DailyProblemResponse,
    HiddenTestCase,
    ProblemRequest,
    ProblemResponse,
)

router = APIRouter()


async def ensure_judge0_healthy() -> None:
    """채점 서버를 쓸 수 없으면 LLM을 부르기 전에 503으로 끝낸다."""
    if not await is_healthy():
        raise JudgeUnavailableError(
            "채점 서버를 사용할 수 없어 문제를 생성하지 않았습니다"
        )


# 이렇게
@router.post("/api/llm/problem")
async def make_problem(
    request: ProblemRequest, http_request: Request
) -> ProblemResponse:
    """
    문제 하나를 생성한다. 검증에서 걸리면 폐기되고 502로 응답한다.

    생성 중에 Spring이 연결을 끊었으면 응답이 닿지 않으므로
    만든 문제를 Spring 저장 API로 따로 보낸다.
    """
    await ensure_judge0_healthy()

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

    problem = build_problem(result)
    if await http_request.is_disconnected():
        await save_undelivered(result.problem_id, problem)
    else:
        await mark_delivered(result.problem_id)

    return ProblemResponse(problem=problem)


@router.post("/api/llm/problem/daily")
async def make_daily_problem() -> DailyProblemResponse:
    """데일리 문제 5개를 생성한다. 하나도 만들지 못하면 502로 응답한다."""
    await ensure_judge0_healthy()

    problems = await generate_daily_problems(Trigger.ON_DEMAND)

    if not problems:
        raise ProblemGenerationError("데일리 문제를 하나도 만들지 못했습니다")

    return DailyProblemResponse(problems=problems)


@router.post("/api/llm/problem/battle")
async def make_battle_problem() -> BattleProblemResponse:
    """배틀 문제 하나를 생성한다. 검증에서 걸리면 폐기되고 502로 응답한다."""
    await ensure_judge0_healthy()

    result = BattleState(**await get_battle_graph().ainvoke(BattleState()))

    if result.is_discarded:
        raise ProblemGenerationError(
            f"배틀 문제를 만들지 못했습니다 "
            f"({result.discard_stage}: {result.discard_reason})"
        )

    return BattleProblemResponse(
        category=result.category,
        problem_title=result.problem_title,
        problem_content=result.problem_content,
        test_cases=[
            HiddenTestCase(input=case.input, output=case.output)
            for case in result.test_cases
        ],
    )
