import json

import httpx
import pytest

from src.client import judge0
from src.client.judge0 import LANGUAGE_TO_ID, run_code
from src.core.enums import ExecutionStatus, Language
from src.schema.problem import ExecutionLimit

LIMIT = ExecutionLimit(
    language=Language.PYTHON, time_limit_ms=1500, memory_limit_kb=262144
)


def fix_judge0(monkeypatch: pytest.MonkeyPatch, handler) -> list[httpx.Request]:
    """judge0 요청을 가짜 전송으로 받는다.

    Returns:
        list[httpx.Request]: 보낸 요청이 쌓인다
    """
    requests: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    def new_client() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url="http://judge0.test", transport=httpx.MockTransport(record)
        )

    monkeypatch.setattr(judge0, "_new_client", new_client)
    return requests


def reply(status_id: int, **fields):
    body = {"status": {"id": status_id, "description": ""}, **fields}
    return lambda request: httpx.Response(201, json=body)


async def run(code: str = "print(1)", stdin: str = ""):
    return await run_code(code, Language.PYTHON, stdin, LIMIT)


@pytest.mark.asyncio
async def test_submits_code_stdin_and_time_limit(monkeypatch) -> None:
    requests = fix_judge0(monkeypatch, reply(3, stdout="11\n"))

    await run("print(sum(map(int, input().split())))", "5 6")

    request = requests[0]
    assert request.url.path == "/submissions"
    assert request.url.params["wait"] == "true"
    assert json.loads(request.content) == {
        "source_code": "print(sum(map(int, input().split())))",
        "language_id": LANGUAGE_TO_ID[Language.PYTHON],
        "stdin": "5 6",
        "cpu_time_limit": 1.5,
    }


@pytest.mark.asyncio
async def test_accepted_is_success(monkeypatch) -> None:
    fix_judge0(monkeypatch, reply(3, stdout="한글 출력\n", stderr=None))

    result = await run()

    assert result.succeeded
    assert result.stdout == "한글 출력\n"
    assert result.stderr == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_id", "expected"),
    [
        (5, ExecutionStatus.TIMED_OUT),
        (6, ExecutionStatus.COMPILE_ERROR),
        (7, ExecutionStatus.RUNTIME_ERROR),
        (11, ExecutionStatus.RUNTIME_ERROR),
        (12, ExecutionStatus.RUNTIME_ERROR),
        (13, ExecutionStatus.INTERNAL_ERROR),
    ],
)
async def test_judge0_status_is_translated(monkeypatch, status_id, expected) -> None:
    fix_judge0(monkeypatch, reply(status_id))

    result = await run()

    assert result.status is expected


@pytest.mark.asyncio
async def test_runtime_error_keeps_stderr(monkeypatch) -> None:
    fix_judge0(monkeypatch, reply(11, stderr="Traceback\nValueError: boom\n"))

    result = await run()

    assert result.stderr.strip().splitlines()[-1] == "ValueError: boom"


@pytest.mark.asyncio
async def test_compile_output_becomes_stderr(monkeypatch) -> None:
    fix_judge0(monkeypatch, reply(6, stderr=None, compile_output="main.cpp:1: error"))

    result = await run_code("int main( {", Language.CPP, "", LIMIT)

    assert result.stderr == "main.cpp:1: error"


@pytest.mark.asyncio
async def test_message_fills_empty_stderr(monkeypatch) -> None:
    fix_judge0(monkeypatch, reply(5, stderr=None, message="Time limit exceeded"))

    result = await run()

    assert result.stderr == "Time limit exceeded"


# ── 채점 서버 장애는 예외 대신 INTERNAL_ERROR ─────────────────────────


@pytest.mark.asyncio
async def test_error_response_is_internal_error(monkeypatch) -> None:
    fix_judge0(monkeypatch, lambda request: httpx.Response(503, text="queue full"))

    result = await run()

    assert result.status is ExecutionStatus.INTERNAL_ERROR
    assert "503" in result.stderr


@pytest.mark.asyncio
async def test_connection_failure_is_internal_error(monkeypatch) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    fix_judge0(monkeypatch, refuse)

    result = await run()

    assert result.status is ExecutionStatus.INTERNAL_ERROR
    assert "ConnectError" in result.stderr


@pytest.mark.asyncio
async def test_non_json_response_is_internal_error(monkeypatch) -> None:
    fix_judge0(monkeypatch, lambda request: httpx.Response(201, text="<html>"))

    result = await run()

    assert result.status is ExecutionStatus.INTERNAL_ERROR
