import httpx

from src.core.config import get_settings
from src.schema import BattleProblem, Problem

_settings = get_settings()

BACKEND_URL = _settings.backend_url


def get_unsolved_problem_count():
    response = httpx.get(f"{BACKEND_URL}/problems/new")

    response.raise_for_status()

    return response.json()


def save_problems(problems: list[Problem]):
    body = {
        "problems": [
            problem.model_dump(mode="json", by_alias=True) for problem in problems
        ],
    }

    response = httpx.post(
        f"{BACKEND_URL}/problems",
        json=body,
    )

    response.raise_for_status()

    return response.json()


def save_daily_problems(problems: list[Problem]):
    body = {
        "problemCount": len(problems),
        "problems": [
            problem.model_dump(mode="json", by_alias=True) for problem in problems
        ],
    }

    response = httpx.post(
        f"{BACKEND_URL}/daily-problems",
        json=body,
    )

    response.raise_for_status()

    return response.json()


def save_battle_problem(battle_problem: BattleProblem):
    response = httpx.post(
        f"{BACKEND_URL}/daily-battles/problem",
        json=battle_problem.model_dump(
            mode="json",
            by_alias=True,
        ),
    )

    response.raise_for_status()

    return response.json()
