import asyncio
import json
import logging

import httpx
import pytest

from src.client import judge0
from src.client.judge0 import LANGUAGE_TO_ID, is_healthy, run_batch, run_code
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


# ── run_batch: 입력마다 제출하고 결과를 모아 폴링 ─────────────────────


def fix_batch(
    monkeypatch: pytest.MonkeyPatch, tokens: list[str], poll
) -> list[httpx.Request]:
    """제출은 tokens를 하나씩 차례로 돌려주고, 조회는 poll의 결과를 돌려준다.

    poll은 조회한 토큰 목록을 받는다.

    Returns:
        list[httpx.Request]: 보낸 요청이 쌓인다
    """
    monkeypatch.setattr(judge0, "POLLING_INTERVAL_S", 0)
    issued = iter(tokens)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(201, json={"token": next(issued)})
        queried = request.url.params["tokens"].split(",")
        return httpx.Response(200, json={"submissions": poll(queried)})

    return fix_judge0(monkeypatch, handler)


def finished(token: str, status_id: int = 3, **fields) -> dict:
    return {"token": token, "status": {"id": status_id, "description": ""}, **fields}


def polls(requests: list[httpx.Request]) -> list[httpx.Request]:
    return [request for request in requests if request.method == "GET"]


def submits(requests: list[httpx.Request]) -> list[httpx.Request]:
    return [request for request in requests if request.method == "POST"]


async def run_many(stdins: list[str]):
    return await run_batch("print(1)", Language.PYTHON, stdins, LIMIT)


@pytest.mark.asyncio
async def test_batch_submits_every_input_with_the_same_code(monkeypatch) -> None:
    requests = fix_batch(
        monkeypatch, ["a", "b"], lambda queried: [finished(t) for t in queried]
    )

    await run_many(["1", "2"])

    payload = {
        "source_code": "print(1)",
        "language_id": LANGUAGE_TO_ID[Language.PYTHON],
        "cpu_time_limit": 1.5,
    }
    sent = submits(requests)
    assert [json.loads(request.content) for request in sent] == [
        payload | {"stdin": "1"},
        payload | {"stdin": "2"},
    ]
    for request in sent:
        assert request.url.path == "/submissions"
        assert request.url.params["wait"] == "false"
        assert request.url.params["base64_encoded"] == "false"


@pytest.mark.asyncio
async def test_batch_does_not_use_the_batch_submit_endpoint(monkeypatch) -> None:
    """dev 채점 서버는 같은 코드를 batch로 제출하면 stdout을 마지막 결과로 덮는다."""
    requests = fix_batch(
        monkeypatch, ["a", "b"], lambda queried: [finished(t) for t in queried]
    )

    await run_many(["1", "2"])

    assert all(r.url.path != "/submissions/batch" for r in submits(requests))


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
    # 두 번째 제출이 검증에서 거절돼 토큰 없이 돌아온다
    replies = iter([{"token": "a"}, {"language_id": ["is not valid"]}])
    requests = fix_judge0(
        monkeypatch, lambda request: httpx.Response(201, json=next(replies))
    )

    results = await run_many(["1", "2"])

    assert [r.status for r in results] == [ExecutionStatus.INTERNAL_ERROR] * 2
    assert polls(requests) == []


@pytest.mark.asyncio
async def test_batch_unexpected_submit_reply_is_internal_error(monkeypatch) -> None:
    """batch 제출 형식(목록)처럼 예상과 다른 응답이 와도 예외를 올리지 않는다."""
    fix_judge0(monkeypatch, lambda request: httpx.Response(201, json=[{"token": "a"}]))

    results = await run_many(["1"])

    assert results[0].status is ExecutionStatus.INTERNAL_ERROR


@pytest.mark.asyncio
async def test_batch_poll_failure_is_internal_error(monkeypatch) -> None:
    monkeypatch.setattr(judge0, "POLLING_INTERVAL_S", 0)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(201, json={"token": "a"})
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


WORKERS_UP = [{"queue": "1.13.1", "available": 4, "idle": 4, "working": 0, "paused": 0}]
TOKEN = "health"


def answer(status: int, body):
    return lambda request: httpx.Response(status, json=body)


def worker_reply(status_id: int, **fields):
    """워커가 실행한 결과를 GET /submissions/batch 응답으로 돌려준다."""
    return answer(200, {"submissions": [finished(TOKEN, status_id, **fields)]})


def fix_health(
    monkeypatch: pytest.MonkeyPatch,
    workers=None,
    api=None,
    submit=None,
    worker=None,
) -> list[httpx.Request]:
    """
    헬스 체크가 보내는 요청에 단계별로 답한다. 넘기지 않은 단계는 정상으로 답한다.

    Parameters:
        workers: GET /workers 응답
        api: wait=true 제출 응답. API 서버가 바로 실행한다
        submit: wait=false 제출 응답. 토큰을 돌려준다
        worker: 결과 조회(GET /submissions/batch) 응답. 워커가 실행한 결과다

    Returns:
        list[httpx.Request]: 보낸 요청이 쌓인다
    """
    workers = workers or answer(200, WORKERS_UP)
    api = api or reply(3, stdout="1\n")
    submit = submit or answer(201, {"token": TOKEN})
    worker = worker or worker_reply(3, stdout="1\n")

    def handle(request: httpx.Request):
        if request.url.path == "/workers":
            return workers(request)
        if request.url.path == "/submissions/batch":
            return worker(request)
        if request.url.params["wait"] == "true":
            return api(request)
        return submit(request)

    monkeypatch.setattr(judge0, "POLLING_INTERVAL_S", 0)
    return fix_judge0(monkeypatch, handle)


def steps(requests: list[httpx.Request]) -> list[str]:
    """보낸 요청을 헬스 체크 단계 이름으로 바꾼다."""
    names = []
    for request in requests:
        if request.url.path == "/workers":
            names.append("workers")
        elif request.url.path == "/submissions/batch":
            names.append("poll")
        elif request.url.params["wait"] == "true":
            names.append("api")
        else:
            names.append("submit")
    return names


def refuse(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused", request=request)


def health_logs(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == judge0.__name__]


@pytest.mark.asyncio
async def test_healthy_when_workers_are_up_and_both_paths_run(monkeypatch) -> None:
    requests = fix_health(monkeypatch)

    assert await is_healthy() is True
    assert steps(requests) == ["workers", "api", "submit", "poll"]
    assert requests[3].url.params["tokens"] == TOKEN


@pytest.mark.asyncio
async def test_no_available_worker_fails_without_submitting(monkeypatch) -> None:
    requests = fix_health(monkeypatch, workers=answer(200, [{"available": 0}]))

    assert await is_healthy() is False
    assert steps(requests) == ["workers"]


@pytest.mark.asyncio
async def test_no_registered_queue_counts_as_no_worker(monkeypatch) -> None:
    # 워커 프로세스가 하나도 없으면 judge0는 빈 목록을 돌려준다
    requests = fix_health(monkeypatch, workers=answer(200, []))

    assert await is_healthy() is False
    assert steps(requests) == ["workers"]


@pytest.mark.asyncio
async def test_workers_are_summed_across_queues(monkeypatch) -> None:
    queues = [{"queue": "a", "available": 0}, {"queue": "b", "available": 2}]
    fix_health(monkeypatch, workers=answer(200, queues))

    assert await is_healthy() is True


@pytest.mark.asyncio
async def test_queue_without_available_field_counts_as_no_worker(monkeypatch) -> None:
    fix_health(monkeypatch, workers=answer(200, [{"queue": "1.13.1"}]))

    assert await is_healthy() is False


@pytest.mark.asyncio
async def test_workers_answer_that_is_not_a_list_fails(monkeypatch) -> None:
    fix_health(monkeypatch, workers=answer(200, {"error": "unauthorized"}))

    assert await is_healthy() is False


@pytest.mark.asyncio
async def test_workers_endpoint_error_fails_without_submitting(monkeypatch) -> None:
    requests = fix_health(monkeypatch, workers=answer(503, {}))

    assert await is_healthy() is False
    assert steps(requests) == ["workers"]


@pytest.mark.asyncio
async def test_api_sandbox_failure_fails_before_trying_the_worker(monkeypatch) -> None:
    # API 서버가 실행하지 못하면 워커까지 볼 필요가 없다
    requests = fix_health(monkeypatch, api=reply(13, message="sandbox failed"))

    assert await is_healthy() is False
    assert steps(requests) == ["workers", "api"]


@pytest.mark.asyncio
async def test_api_server_error_fails(monkeypatch) -> None:
    fix_health(monkeypatch, api=answer(500, {}))

    assert await is_healthy() is False


@pytest.mark.asyncio
async def test_worker_failure_fails_even_when_api_runs(monkeypatch) -> None:
    # 등록만 남은 워커나 워커 쪽 샌드박스 장애. wait=true만 보면 놓친다
    fix_health(monkeypatch, worker=worker_reply(13, message="sandbox failed"))

    assert await is_healthy() is False


@pytest.mark.asyncio
async def test_worker_that_never_finishes_fails(monkeypatch) -> None:
    # 워커가 멈춰 있으면 제출이 대기(1) 상태에서 끝나지 않는다
    monkeypatch.setattr(judge0, "HEALTH_CHECK_TIMEOUT_S", 0.05)
    fix_health(monkeypatch, worker=worker_reply(1))
    monkeypatch.setattr(judge0, "POLLING_INTERVAL_S", 0.01)

    assert await is_healthy() is False


@pytest.mark.asyncio
async def test_worker_result_missing_after_polling_deadline_fails(monkeypatch) -> None:
    # 헬스 체크 제한보다 조회 한도가 먼저 끝나도 실패로 본다
    monkeypatch.setattr(judge0, "POLLING_DEADLINE_S", 0.05)
    fix_health(monkeypatch, worker=answer(200, {"submissions": [None]}))
    monkeypatch.setattr(judge0, "POLLING_INTERVAL_S", 0.01)

    assert await is_healthy() is False


@pytest.mark.asyncio
async def test_worker_submit_without_token_fails(monkeypatch) -> None:
    requests = fix_health(monkeypatch, submit=answer(201, {"error": "queue full"}))

    assert await is_healthy() is False
    assert "poll" not in steps(requests)


@pytest.mark.asyncio
async def test_worker_poll_error_fails(monkeypatch) -> None:
    fix_health(monkeypatch, worker=answer(500, {}))

    assert await is_healthy() is False


@pytest.mark.asyncio
async def test_connection_failure_fails(monkeypatch) -> None:
    fix_judge0(monkeypatch, refuse)

    assert await is_healthy() is False


@pytest.mark.asyncio
async def test_timeout_covers_every_step_together(monkeypatch) -> None:
    # 요청 하나하나는 제한 안에 끝나도 합치면 넘는다
    async def slow_workers(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.15)
        return httpx.Response(200, json=WORKERS_UP)

    async def slow_api(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.15)
        return httpx.Response(201, json={"status": {"id": 3}})

    monkeypatch.setattr(judge0, "HEALTH_CHECK_TIMEOUT_S", 0.2)
    fix_health(monkeypatch, workers=slow_workers, api=slow_api)

    assert await is_healthy() is False


@pytest.mark.asyncio
async def test_failure_is_logged_in_one_line_without_traceback(
    monkeypatch, caplog
) -> None:
    fix_judge0(monkeypatch, refuse)

    with caplog.at_level(logging.WARNING, logger=judge0.__name__):
        await is_healthy()

    [record] = health_logs(caplog)
    assert record.levelno == logging.WARNING
    assert record.exc_info is None
    assert "ConnectError" in record.getMessage()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failing", "where"),
    [
        ({"api": reply(13, message="sandbox failed")}, "API 서버"),
        ({"worker": worker_reply(13, message="sandbox failed")}, "워커"),
    ],
)
async def test_failed_run_logs_where_it_failed(
    monkeypatch, caplog, failing, where
) -> None:
    fix_health(monkeypatch, **failing)

    with caplog.at_level(logging.WARNING, logger=judge0.__name__):
        await is_healthy()

    [record] = health_logs(caplog)
    message = record.getMessage()
    assert where in message
    assert ExecutionStatus.INTERNAL_ERROR.value in message
    assert "sandbox failed" in message


@pytest.mark.asyncio
async def test_healthy_check_logs_nothing(monkeypatch, caplog) -> None:
    fix_health(monkeypatch)

    with caplog.at_level(logging.DEBUG, logger=judge0.__name__):
        await is_healthy()

    assert health_logs(caplog) == []
