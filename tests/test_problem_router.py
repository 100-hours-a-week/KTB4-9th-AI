import pytest
from fastapi.testclient import TestClient

from src.core.enums import ErrorCode
from src.main import app
from src.router import problem as module

ROUTES = [
    ("/api/llm/problem", {"difficulty": "LV1", "category": "DP"}),
    ("/api/llm/problem/daily", None),
    ("/api/llm/problem/battle", None),
]


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def judge0_down(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """
    judge0를 죽은 상태로 두고 생성 진입점이 불리는지 기록한다.

    Returns:
        list[str]: 불린 생성 진입점 이름이 쌓인다
    """
    started: list[str] = []

    async def unhealthy() -> bool:
        return False

    def record(name: str):
        def start(*args, **kwargs):
            started.append(name)
            raise AssertionError(f"{name}이 불리면 안 된다")

        return start

    monkeypatch.setattr(module, "is_healthy", unhealthy)
    monkeypatch.setattr(module, "get_graph", record("get_graph"))
    monkeypatch.setattr(module, "get_battle_graph", record("get_battle_graph"))
    monkeypatch.setattr(
        module, "generate_daily_problems", record("generate_daily_problems")
    )
    return started


@pytest.mark.parametrize(("path", "body"), ROUTES)
def test_unhealthy_judge0_answers_503_before_generating(
    client, judge0_down, path, body
) -> None:
    response = client.post(path, json=body)

    assert response.status_code == 503
    assert response.json()["errorCode"] == ErrorCode.JUDGE_UNAVAILABLE
    assert judge0_down == []


def test_invalid_request_does_not_reach_the_health_check(client, monkeypatch) -> None:
    async def fail() -> bool:
        raise AssertionError("검증 실패한 요청은 judge0를 부르면 안 된다")

    monkeypatch.setattr(module, "is_healthy", fail)

    response = client.post(
        "/api/llm/problem", json={"difficulty": "LV9", "category": "DP"}
    )

    assert response.status_code == 422
