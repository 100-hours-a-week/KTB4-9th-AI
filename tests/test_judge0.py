import json

import httpx
import pytest

from src.client import judge0
from src.client.judge0 import LANGUAGE_TO_ID, run_batch, run_code
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


# ── run_batch: 여러 입력을 한 번에 제출하고 결과를 폴링 ───────────────


def fix_batch(
    monkeypatch: pytest.MonkeyPatch, tokens: list[str], poll
) -> list[httpx.Request]:
    """batch 제출은 tokens를 돌려주고, 조회는 poll(조회한 토큰 목록)의 결과를 돌려준다.

    Returns:
        list[httpx.Request]: 보낸 요청이 쌓인다
    """
    monkeypatch.setattr(judge0, "POLLING_INTERVAL_S", 0)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(201, json=[{"token": token} for token in tokens])
        queried = request.url.params["tokens"].split(",")
        return httpx.Response(200, json={"submissions": poll(queried)})

    return fix_judge0(monkeypatch, handler)


def finished(token: str, status_id: int = 3, **fields) -> dict:
    return {"token": token, "status": {"id": status_id, "description": ""}, **fields}


def polls(requests: list[httpx.Request]) -> list[httpx.Request]:
    return [request for request in requests if request.method == "GET"]


async def run_many(stdins: list[str]):
    return await run_batch("print(1)", Language.PYTHON, stdins, LIMIT)


@pytest.mark.asyncio
async def test_batch_submits_every_input_with_the_same_code(monkeypatch) -> None:
    requests = fix_batch(
        monkeypatch, ["a", "b"], lambda queried: [finished(t) for t in queried]
    )

    await run_many(["1", "2"])

    submit = requests[0]
    assert submit.url.path == "/submissions/batch"
    assert submit.url.params["base64_encoded"] == "false"
    payload = {
        "source_code": "print(1)",
        "language_id": LANGUAGE_TO_ID[Language.PYTHON],
        "cpu_time_limit": 1.5,
    }
    assert json.loads(submit.content) == {
        "submissions": [payload | {"stdin": "1"}, payload | {"stdin": "2"}]
    }


@pytest.mark.asyncio
async def test_batch_polls_until_every_submission_finishes(monkeypatch) -> None:
    rounds = iter([2, 3])  # 첫 조회는 실행 중, 다음 조회에서 끝난다

    def poll(queried: list[str]) -> list[dict]:
        status_id = next(rounds)
        return [finished(t, status_id) for t in queried]

    requests = fix_batch(monkeypatch, ["a", "b"], poll)

    results = await run_many(["1", "2"])

    assert [result.status for result in results] == [ExecutionStatus.SUCCEEDED] * 2
    assert len(polls(requests)) == 2


@pytest.mark.asyncio
async def test_batch_queries_only_unfinished_tokens(monkeypatch) -> None:
    rounds = iter([{"a": 3, "b": 2}, {"b": 3}])  # a가 먼저 끝나고 b가 뒤에 끝난다

    def poll(queried: list[str]) -> list[dict]:
        status_ids = next(rounds)
        return [finished(t, status_ids[t]) for t in queried]

    requests = fix_batch(monkeypatch, ["a", "b"], poll)

    await run_many(["1", "2"])

    assert [r.url.params["tokens"] for r in polls(requests)] == ["a,b", "b"]


@pytest.mark.asyncio
async def test_batch_keeps_input_order(monkeypatch) -> None:
    fix_batch(
        monkeypatch,
        ["a", "b", "c"],
        lambda queried: [finished(t, stdout=t) for t in reversed(queried)],
    )

    results = await run_many(["1", "2", "3"])

    assert [result.stdout for result in results] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_batch_translates_each_status(monkeypatch) -> None:
    fix_batch(
        monkeypatch,
        ["a", "b"],
        lambda queried: [finished("a"), finished("b", 11, stderr="boom")],
    )

    results = await run_many(["1", "2"])

    assert [result.status for result in results] == [
        ExecutionStatus.SUCCEEDED,
        ExecutionStatus.RUNTIME_ERROR,
    ]
    assert results[1].stderr == "boom"


@pytest.mark.asyncio
async def test_batch_with_no_input_sends_nothing(monkeypatch) -> None:
    requests = fix_batch(monkeypatch, [], lambda queried: [])

    assert await run_many([]) == []
    assert requests == []


# ── run_batch: 채점 서버 장애와 대기 초과는 입력마다 INTERNAL_ERROR ───────


@pytest.mark.asyncio
async def test_batch_submit_failure_is_internal_error(monkeypatch) -> None:
    fix_judge0(monkeypatch, lambda request: httpx.Response(503, text="queue full"))

    results = await run_many(["1", "2"])

    assert len(results) == 2
    assert all(r.status is ExecutionStatus.INTERNAL_ERROR for r in results)
    assert all("503" in r.stderr for r in results)


@pytest.mark.asyncio
async def test_batch_missing_token_is_internal_error(monkeypatch) -> None:
    rejected = [{"token": "a"}, {"language_id": ["is not valid"]}]
    requests = fix_judge0(
        monkeypatch, lambda request: httpx.Response(201, json=rejected)
    )

    results = await run_many(["1", "2"])

    assert [r.status for r in results] == [ExecutionStatus.INTERNAL_ERROR] * 2
    assert polls(requests) == []


@pytest.mark.asyncio
async def test_batch_poll_failure_is_internal_error(monkeypatch) -> None:
    monkeypatch.setattr(judge0, "POLLING_INTERVAL_S", 0)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(201, json=[{"token": "a"}])
        return httpx.Response(500)

    fix_judge0(monkeypatch, handler)

    results = await run_many(["1"])

    assert results[0].status is ExecutionStatus.INTERNAL_ERROR
    assert "500" in results[0].stderr


@pytest.mark.asyncio
async def test_batch_gives_up_on_unfinished_after_the_deadline(monkeypatch) -> None:
    monkeypatch.setattr(judge0, "POLLING_DEADLINE_S", 0.05)
    fix_batch(
        monkeypatch,
        ["a", "b"],
        lambda queried: [finished(t, 3 if t == "a" else 1) for t in queried],
    )

    results = await run_many(["1", "2"])

    assert results[0].status is ExecutionStatus.SUCCEEDED
    assert results[1].status is ExecutionStatus.INTERNAL_ERROR
    assert "대기 시간 초과" in results[1].stderr


@pytest.mark.asyncio
async def test_batch_skips_unknown_tokens_until_the_deadline(monkeypatch) -> None:
    # judge0는 찾지 못한 토큰 자리에 null을 넣는다
    monkeypatch.setattr(judge0, "POLLING_DEADLINE_S", 0.05)
    fix_batch(monkeypatch, ["a"], lambda queried: [None])

    results = await run_many(["1"])

    assert results[0].status is ExecutionStatus.INTERNAL_ERROR
