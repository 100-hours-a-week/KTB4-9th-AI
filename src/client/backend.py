"""Spring 백엔드 클라이언트.

새벽 배치가 쓴다. 01:00에 조합별 재고를 묻고, 03:00에 만든 문제를 보낸다.
실패는 예외로 올린다. 재시도할지, 다음 날로 미룰지는 배치가 정한다.

재고 조회와 저장은 Spring이 제공하는 API라 Spring 형식에 맞춘다.
우리 형식과 다른 점은 이 모듈에서만 바꾸고, 안쪽(LLM·DB·우리 API)은 그대로 둔다.
- enum은 값이 아니라 이름(대문자)으로 주고받는다. 예: python → PYTHON, str → STRING
- 카테고리 HASH_TABLE은 Spring에서 HASH다
- 필드 이름: problemContent → problemDescription, solutionCodes → hintSolutionCodes
"""

import logging

import httpx
from pydantic import ValidationError

from src.core.config import get_settings
from src.core.enums import Category
from src.schema.batch import ProblemDemand, ProblemDemandResponse
from src.schema.problem import BattleProblem, Problem

logger = logging.getLogger(__name__)

_settings = get_settings()

BACKEND_URL = _settings.backend_url

REQUEST_TIMEOUT_S = 30.0

# 이름이 다른 카테고리만 적는다. 나머지는 enum 이름이 Spring과 같다.
CATEGORY_TO_SPRING = {Category.HASH_TABLE: "HASH"}
CATEGORY_FROM_SPRING = {name: category for category, name in CATEGORY_TO_SPRING.items()}


def _new_client() -> httpx.AsyncClient:
    """Spring HTTP 클라이언트. 테스트에서 가짜 전송으로 바꿔 끼운다."""
    return httpx.AsyncClient(base_url=BACKEND_URL, timeout=REQUEST_TIMEOUT_S)


def _spring_category(category: Category) -> str:
    return CATEGORY_TO_SPRING.get(category, category.name)


def _problem_body(problem: Problem) -> dict:
    """
    문제를 Spring 저장 형식(AiProblemsCreateRequestDto)으로 바꾼다.

    Spring이 읽지 않는 필드(requestedDifficulty, dataCount)는 보내지 않는다.
    """
    return {
        "problemTitle": problem.problem_title,
        "problemDescription": problem.problem_content,
        "inputFormat": problem.input_format,
        "outputFormat": problem.output_format,
        "difficulty": problem.difficulty.name,
        "category": _spring_category(problem.category),
        "categorySelectReason": problem.category_select_reason,
        "solutionKeywords": problem.solution_keywords,
        "problemExamples": [
            example.model_dump(mode="json") for example in problem.problem_examples
        ],
        "inputConstraints": [
            {
                "target": constraint.target,
                "scope": constraint.scope.name,
                "dataType": constraint.data_type.name,
                "minValue": constraint.min_value,
                "maxValue": constraint.max_value,
                "specialConditions": constraint.special_conditions,
            }
            for constraint in problem.input_constraints
        ],
        "executionLimits": [
            {
                "language": limit.language.name,
                "timeLimitMs": limit.time_limit_ms,
                "memoryLimitKb": limit.memory_limit_kb,
            }
            for limit in problem.execution_limits
        ],
        "hiddenTestCases": [
            case.model_dump(mode="json") for case in problem.hidden_test_cases
        ],
        "hintComments": [
            {"language": hint.language.name, "content": hint.content}
            for hint in problem.hint_comments
        ],
        "hintSolutionCodes": [
            {"language": code.language.name, "content": code.content}
            for code in problem.solution_codes
        ],
    }


def _battle_body(problem: BattleProblem) -> dict:
    """배틀 문제를 Spring 저장 형식(AiBattleCreateRequestDto)으로 바꾼다."""
    return {
        "category": _spring_category(problem.category),
        "problemTitle": problem.problem_title,
        "problemDescription": problem.problem_content,
        "testCases": [case.model_dump(mode="json") for case in problem.test_cases],
    }


def _from_spring_demand(entry: dict) -> dict:
    """재고 항목의 Spring 카테고리 이름을 우리 값으로 바꾼다."""
    category = entry.get("category")
    if category in CATEGORY_FROM_SPRING:
        return {**entry, "category": CATEGORY_FROM_SPRING[category]}
    return entry


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
            demands.append(ProblemDemand.model_validate(_from_spring_demand(entry)))
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
    body = {"problems": [_problem_body(problem) for problem in problems]}
    async with _new_client() as client:
        response = await client.post("/problems", json=body)
        response.raise_for_status()


async def save_daily_problems(problems: list[Problem]) -> None:
    """
    데일리 문제를 저장 요청한다.

    Raises:
        httpx.HTTPError: 연결 실패나 2xx가 아닌 응답
    """
    body = {
        "problemCount": len(problems),
        "problems": [_problem_body(problem) for problem in problems],
    }
    async with _new_client() as client:
        response = await client.post("/daily-problems", json=body)
        response.raise_for_status()


async def save_battle_problem(battle_problem: BattleProblem) -> None:
    """
    배틀 문제를 저장 요청한다. Spring은 하루 한 문제를 그날의 배틀로 쓴다.

    Raises:
        httpx.HTTPError: 연결 실패나 2xx가 아닌 응답
    """
    async with _new_client() as client:
        response = await client.post(
            "/daily-battles/problem", json=_battle_body(battle_problem)
        )
        response.raise_for_status()
