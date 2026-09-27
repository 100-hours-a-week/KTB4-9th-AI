import json

import httpx
import pytest

from src.client import backend
from src.client.backend import (
    get_problem_demands,
    save_battle_problem,
    save_daily_problems,
    save_problems,
)
from src.core.enums import (
    Category,
    ConstraintDataType,
    ConstraintScope,
    Difficulty,
    Language,
)
from src.schema.batch import ProblemDemand
from src.schema.problem import (
    BattleProblem,
    ExecutionLimit,
    HiddenTestCase,
    HintComment,
    InputConstraint,
    Problem,
    ProblemExample,
    SolutionCode,
)

# Spring enum 이름 (KTB4-9th-BE common 패키지). 계약이 바뀌면 여기부터 고친다.
SPRING_CATEGORIES = {
    "ARRAY", "STRING", "DP", "GRAPH", "TREE", "STACK_QUEUE", "BINARY_SEARCH",
    "GREEDY", "BACKTRACKING", "TWO_POINTER", "HASH", "HEAP", "SORTING",
    "IMPLEMENTATION", "BRUTE_FORCE", "MATH",
}  # fmt: skip
SPRING_LANGUAGES = {"PYTHON", "JAVA", "JAVASCRIPT", "CPP"}
SPRING_SCOPES = {"INPUT", "OUTPUT"}
SPRING_DATATYPES = {"INT", "LONG", "FLOAT", "DOUBLE", "STRING", "CHAR", "BOOLEAN"}

# Spring 저장 DTO가 읽는 필드 (AiProblemsCreateRequestDto.ProblemsInfo)
SPRING_PROBLEM_FIELDS = {
    "problemTitle",
    "problemDescription",
    "inputFormat",
    "outputFormat",
    "difficulty",
    "category",
    "categorySelectReason",
    "solutionKeywords",
    "problemExamples",
    "inputConstraints",
    "executionLimits",
    "hiddenTestCases",
    "hintComments",
    "hintSolutionCodes",
}


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


def make_full_problem() -> Problem:
    """Spring으로 가는 모든 필드가 채워진 문제."""
    return make_problem().model_copy(
        update={
            "input_constraints": [
                InputConstraint(
                    target="S",
                    scope=ConstraintScope.INPUT,
                    data_type=ConstraintDataType.STRING,
                    data_count=1,
                ),
                InputConstraint(
                    target="output",
                    scope=ConstraintScope.OUTPUT,
                    data_type=ConstraintDataType.BOOLEAN,
                ),
            ],
            "execution_limits": [
                ExecutionLimit(
                    language=lang, time_limit_ms=1000, memory_limit_kb=262144
                )
                for lang in Language
            ],
            "problem_examples": [ProblemExample(input="3 5", output="1")],
            "solution_keywords": ["해시맵"],
            "hidden_test_cases": [HiddenTestCase(input="1 2", output="0")],
            "hint_comments": [
                HintComment(language=lang, content="# 힌트") for lang in Language
            ],
            "solution_codes": [
                SolutionCode(language=lang, content="print(1)") for lang in Language
            ],
        }
    )


def make_battle() -> BattleProblem:
    return BattleProblem(
        category=Category.HASH_TABLE,
        problem_title="중복 찾기",
        problem_content="처음으로 두 번 나온 수를 출력하라",
        test_cases=[HiddenTestCase(input=f"{n}", output=f"{n}") for n in range(3)],
    )


def posted_problem(requests: list[httpx.Request]) -> dict:
    return json.loads(requests[0].content)["problems"][0]


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
async def test_spring_hash_category_is_read_as_hash_table(monkeypatch) -> None:
    """못 읽고 건너뛰면 HASH_TABLE 재고가 늘 0이라 매일 15개씩 더 만든다."""
    fix_backend(
        monkeypatch,
        demands_reply({"difficulty": "LV1", "category": "HASH", "count": 3}),
    )

    demands = await get_problem_demands()

    assert demands == [
        ProblemDemand(difficulty=Difficulty.LV1, category=Category.HASH_TABLE, count=3)
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
async def test_problem_fields_are_what_spring_reads(monkeypatch) -> None:
    requests = fix_backend(monkeypatch, lambda request: httpx.Response(201))

    await save_problems([make_full_problem()])

    body = posted_problem(requests)
    assert set(body) == SPRING_PROBLEM_FIELDS
    assert body["problemDescription"] == "합이 M인 쌍의 수를 구하라"
    assert [c["content"] for c in body["hintSolutionCodes"]] == ["print(1)"] * 4


@pytest.mark.asyncio
async def test_enums_are_sent_by_spring_name(monkeypatch) -> None:
    """Spring은 enum 이름(대문자)만 읽는다. 값(python, str)을 보내면 500이 난다."""
    requests = fix_backend(monkeypatch, lambda request: httpx.Response(201))

    await save_problems([make_full_problem()])

    body = posted_problem(requests)
    assert body["category"] == "HASH"
    assert body["difficulty"] == "LV2"
    assert [c["scope"] for c in body["inputConstraints"]] == ["INPUT", "OUTPUT"]
    assert [c["dataType"] for c in body["inputConstraints"]] == ["STRING", "BOOLEAN"]
    for key in ("executionLimits", "hintComments", "hintSolutionCodes"):
        assert [item["language"] for item in body[key]] == [
            "PYTHON",
            "JAVA",
            "JAVASCRIPT",
            "CPP",
        ]


def test_every_enum_has_a_spring_name() -> None:
    """우리 enum에 값을 더하면 Spring에 같은 이름이 있는지 여기서 걸린다."""
    real = [c for c in Category if c != Category.RANDOM]
    assert {backend._spring_category(c) for c in real} == SPRING_CATEGORIES
    assert {lang.name for lang in Language} == SPRING_LANGUAGES
    assert {scope.name for scope in ConstraintScope} == SPRING_SCOPES
    assert {dtype.name for dtype in ConstraintDataType} <= SPRING_DATATYPES


@pytest.mark.asyncio
async def test_battle_is_posted_in_spring_form(monkeypatch) -> None:
    requests = fix_backend(monkeypatch, lambda request: httpx.Response(201))

    await save_battle_problem(make_battle())

    request = requests[0]
    body = json.loads(request.content)
    assert request.url.path == "/daily-battles/problem"
    assert body == {
        "category": "HASH",
        "problemTitle": "중복 찾기",
        "problemDescription": "처음으로 두 번 나온 수를 출력하라",
        "testCases": [{"input": f"{n}", "output": f"{n}"} for n in range(3)],
    }


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
    with pytest.raises(httpx.HTTPStatusError):
        await save_battle_problem(make_battle())


@pytest.mark.asyncio
async def test_connection_failure_is_raised(monkeypatch) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    fix_backend(monkeypatch, refuse)

    with pytest.raises(httpx.ConnectError):
        await save_problems([make_problem()])
