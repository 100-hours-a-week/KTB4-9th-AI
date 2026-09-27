import json

import httpx
import pytest

from src.client import backend
from src.client.backend import (
    get_problem_demands,
    save_daily_problems,
    save_problems,
)
from src.core.enums import Category, Difficulty
from src.schema.batch import ProblemDemand
from src.schema.problem import Problem


def fix_backend(monkeypatch: pytest.MonkeyPatch, handler) -> list[httpx.Request]:
    """Spring 요청을 가짜 전송으로 받는다.

    Returns:
        list[httpx.Request]: 보낸 요청이 쌓인다
    """
    requests: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    def new_client() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url="http://backend.test", transport=httpx.MockTransport(record)
        )

    monkeypatch.setattr(backend, "_new_client", new_client)
    return requests


def demands_reply(*entries: dict):
    body = {
        "message": "아무도 풀지 않은 문제 조회에 성공했습니다.",
        "data": {"problemCounts": list(entries)},
    }
    return lambda request: httpx.Response(200, json=body)


def make_problem() -> Problem:
    return Problem(
        problem_title="두 수의 합",
        problem_content="합이 M인 쌍의 수를 구하라",
        input_format="N M",
        output_format="쌍의 수",
        requested_difficulty=Difficulty.LV2,
        difficulty=Difficulty.LV2,
        category=Category.HASH_TABLE,
        category_select_reason="해시맵으로 푼다",
        input_constraints=[],
        execution_limits=[],
        problem_examples=[],
    )


# ── 재고 조회 ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_demands_are_parsed_into_enums(monkeypatch) -> None:
    requests = fix_backend(
        monkeypatch,
        demands_reply(
            {"difficulty": "LV2", "category": "DP", "count": 3},
            {"difficulty": "LV1", "category": "GREEDY", "count": 0},
        ),
    )

    demands = await get_problem_demands()

    assert requests[0].method == "GET"
    assert requests[0].url.path == "/problems/new"
    assert demands == [
        ProblemDemand(difficulty=Difficulty.LV2, category=Category.DP, count=3),
        ProblemDemand(difficulty=Difficulty.LV1, category=Category.GREEDY, count=0),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "broken",
    [
        {"difficulty": "LV2", "category": "Array", "count": 1},  # 표기가 다름
        {"difficulty": "LV9", "category": "DP", "count": 1},  # 없는 난이도
        {"difficulty": "LV2", "category": "RANDOM", "count": 1},  # 조합이 아님
        {"difficulty": "LV2", "category": "DP", "count": -1},  # 음수
        {"difficulty": "LV2", "category": "DP"},  # 개수 없음
    ],
)
async def test_unreadable_entry_is_skipped_not_fatal(monkeypatch, broken) -> None:
    fix_backend(
        monkeypatch,
        demands_reply(broken, {"difficulty": "LV3", "category": "ARRAY", "count": 2}),
    )

    demands = await get_problem_demands()

    assert [(d.difficulty, d.category) for d in demands] == [
        (Difficulty.LV3, Category.ARRAY)
    ]


@pytest.mark.asyncio
async def test_empty_demands_is_an_empty_list(monkeypatch) -> None:
    fix_backend(monkeypatch, demands_reply())

    assert await get_problem_demands() == []


@pytest.mark.asyncio
async def test_demand_server_error_is_raised(monkeypatch) -> None:
    fix_backend(monkeypatch, lambda request: httpx.Response(500))

    with pytest.raises(httpx.HTTPStatusError):
        await get_problem_demands()


@pytest.mark.asyncio
async def test_unexpected_envelope_is_raised(monkeypatch) -> None:
    """바깥 구조가 바뀌면 항목 단위로 건너뛸 수 없다. 배치가 알아야 한다."""
    fix_backend(
        monkeypatch,
        lambda request: httpx.Response(200, json={"data": [{"count": 1}]}),
    )

    with pytest.raises(ValueError):
        await get_problem_demands()


# ── 저장 요청 ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_problems_are_posted_in_camel_case(monkeypatch) -> None:
    requests = fix_backend(monkeypatch, lambda request: httpx.Response(201))

    await save_problems([make_problem(), make_problem()])

    request = requests[0]
    body = json.loads(request.content)
    assert request.method == "POST"
    assert request.url.path == "/problems"
    assert list(body) == ["problems"]
    assert len(body["problems"]) == 2
    assert body["problems"][0]["problemTitle"] == "두 수의 합"
    assert "problem_title" not in body["problems"][0]


@pytest.mark.asyncio
async def test_daily_problems_carry_their_count(monkeypatch) -> None:
    requests = fix_backend(monkeypatch, lambda request: httpx.Response(201))

    await save_daily_problems([make_problem()] * 3)

    request = requests[0]
    body = json.loads(request.content)
    assert request.url.path == "/daily-problems"
    assert body["problemCount"] == 3
    assert len(body["problems"]) == 3


@pytest.mark.asyncio
async def test_success_without_a_json_body_is_still_success(monkeypatch) -> None:
    """저장은 됐는데 본문 해석에서 실패하면 다음 날 같은 문제가 또 나간다."""
    fix_backend(monkeypatch, lambda request: httpx.Response(200, text="OK"))

    await save_problems([make_problem()])
    await save_daily_problems([make_problem()])


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 500, 503])
async def test_save_failure_is_raised(monkeypatch, status) -> None:
    fix_backend(monkeypatch, lambda request: httpx.Response(status))

    with pytest.raises(httpx.HTTPStatusError):
        await save_problems([make_problem()])
    with pytest.raises(httpx.HTTPStatusError):
        await save_daily_problems([make_problem()])


@pytest.mark.asyncio
async def test_connection_failure_is_raised(monkeypatch) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    fix_backend(monkeypatch, refuse)

    with pytest.raises(httpx.ConnectError):
        await save_problems([make_problem()])
