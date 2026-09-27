"""Spring 백엔드 클라이언트.

새벽 배치가 쓴다. 01:00에 조합별 재고를 묻고, 03:00에 만든 문제를 보낸다.
실패는 예외로 올린다. 재시도할지, 다음 날로 미룰지는 배치가 정한다.
"""

import logging

import httpx
from pydantic import ValidationError

from src.core.config import get_settings
from src.schema.batch import ProblemDemand, ProblemDemandResponse
from src.schema.problem import BattleProblem, Problem

logger = logging.getLogger(__name__)

_settings = get_settings()

BACKEND_URL = _settings.backend_url

REQUEST_TIMEOUT_S = 30.0


def _new_client() -> httpx.AsyncClient:
    """Spring HTTP 클라이언트. 테스트에서 가짜 전송으로 바꿔 끼운다."""
    return httpx.AsyncClient(base_url=BACKEND_URL, timeout=REQUEST_TIMEOUT_S)


def _dump(problems: list[Problem]) -> list[dict]:
    return [problem.model_dump(mode="json", by_alias=True) for problem in problems]


async def get_problem_demands() -> list[ProblemDemand]:
    """
    조합(난이도, 카테고리)별로 아직 아무도 풀지 않은 문제 수를 받는다.

    해석할 수 없는 항목은 경고만 남기고 건너뛴다. 한 항목 때문에
    그날 배치 전체를 멈출 이유는 없다.

    Returns:
        list[ProblemDemand]: 해석한 항목들

    Raises:
        httpx.HTTPError: 연결 실패나 2xx가 아닌 응답
        ValueError: 본문이 JSON이 아니거나 바깥 구조(data.problemCounts)가 다른 경우
    """
    async with _new_client() as client:
        response = await client.get("/problems/new")
        response.raise_for_status()

    body = ProblemDemandResponse.model_validate(response.json())

    demands: list[ProblemDemand] = []
    for entry in body.data.problem_counts:
        try:
            demands.append(ProblemDemand.model_validate(entry))
        except ValidationError as error:
            logger.warning("재고 항목을 해석할 수 없어 건너뜀: %r (%s)", entry, error)
    return demands


async def save_problems(problems: list[Problem]) -> None:
    """
    일반 문제를 저장 요청한다.

    응답 본문은 읽지 않는다. Spring이 저장한 뒤 본문 해석에서 실패하면
    전송 완료 표시를 못 해 다음 날 같은 문제가 또 나가기 때문이다.

    Raises:
        httpx.HTTPError: 연결 실패나 2xx가 아닌 응답
    """
    async with _new_client() as client:
        response = await client.post("/problems", json={"problems": _dump(problems)})
        response.raise_for_status()


async def save_daily_problems(problems: list[Problem]) -> None:
    """
    데일리 문제를 저장 요청한다.

    Raises:
        httpx.HTTPError: 연결 실패나 2xx가 아닌 응답
    """
    body = {"problemCount": len(problems), "problems": _dump(problems)}
    async with _new_client() as client:
        response = await client.post("/daily-problems", json=body)
        response.raise_for_status()


async def save_battle_problem(battle_problem: BattleProblem) -> None:
    """
    배틀 문제를 저장 요청한다. 배틀 생성 그래프가 생기면 쓴다.

    Raises:
        httpx.HTTPError: 연결 실패나 2xx가 아닌 응답
    """
    body = battle_problem.model_dump(mode="json", by_alias=True)
    async with _new_client() as client:
        response = await client.post("/daily-battles/problem", json=body)
        response.raise_for_status()
