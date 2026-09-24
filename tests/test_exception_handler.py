import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.api import app
from src.core.enums import ErrorCode
from src.core.exception import CosmosError, ProblemGenerationError


@app.get("/_test/cosmos-error")
async def _raise_cosmos_error():
    raise ProblemGenerationError("모델이 응답하지 않음")


@app.get("/_test/unexpected")
async def _raise_unexpected():
    raise RuntimeError("예상 못한 오류")


@app.get("/_test/http-404")
async def _raise_http_404():
    raise HTTPException(404, "없는 문제")


@pytest.fixture
def client():
    # raise_server_exceptions=False라야 미처리 예외도 응답으로 받아볼 수 있다
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_cosmos_error_maps_to_its_own_code_and_status(client) -> None:
    response = client.get("/_test/cosmos-error")

    assert response.status_code == 502
    assert response.json() == {
        "success": False,
        "message": "모델이 응답하지 않음",
        "errorCode": ErrorCode.PROBLEM_GENERATION_FAILED,
    }


def test_unexpected_error_becomes_500_without_leaking_detail(client) -> None:
    response = client.get("/_test/unexpected")
    body = response.json()

    assert response.status_code == 500
    assert body["errorCode"] == ErrorCode.INTERNAL_SERVER_ERROR
    assert "예상 못한 오류" not in body["message"]  # 내부 사정을 밖으로 흘리지 않는다


def test_invalid_enum_value_maps_to_field_code(client) -> None:
    response = client.post(
        "/api/llm/problem", json={"difficulty": "LV9", "category": "DP"}
    )

    assert response.status_code == 422
    assert response.json()["errorCode"] == ErrorCode.INVALID_DIFFICULTY


def test_missing_field_maps_to_missing_code(client) -> None:
    response = client.post("/api/llm/problem", json={"category": "DP"})

    assert response.status_code == 422
    assert response.json()["errorCode"] == ErrorCode.MISSING_REQUIRED_FIELD


def test_client_error_keeps_status_and_omits_error_code(client) -> None:
    response = client.get("/_test/http-404")

    assert response.status_code == 404
    assert response.json()["errorCode"] is None


def test_every_error_response_shares_base_response_shape(client) -> None:
    paths = ["/_test/cosmos-error", "/_test/unexpected", "/_test/http-404"]

    for path in paths:
        body = client.get(path).json()
        assert set(body) == {"success", "message", "errorCode"}, path
        assert body["success"] is False, path


def test_cosmos_error_defaults_message_to_its_code() -> None:
    class NamelessError(CosmosError):
        pass

    assert NamelessError().message == ErrorCode.INTERNAL_SERVER_ERROR
