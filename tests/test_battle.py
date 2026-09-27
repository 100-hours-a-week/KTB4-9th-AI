import pytest

from src.client.judge0 import RunResult
from src.core.enums import Category, DiscardReason, ExecutionStatus
from src.problem.battle import nodes as module
from src.problem.battle.nodes import (
    GeneratedBattleProblem,
    build_prompt,
    fill_outputs,
    generate_battle_problem,
    static_validate,
)
from src.problem.battle.state import (
    MAX_CONTENT_LENGTH,
    MAX_TITLE_LENGTH,
    TEST_CASE_COUNT,
    BattleState,
    BattleTestCase,
)

VALID_PROBLEM = {
    "problem_title": "기준 이상인 첫 위치",
    "problem_content": "정렬된 수열에서 K 이상인 첫 원소의 위치를 구하시오.",
    "algorithm_core": "정렬된 배열에서 이분 탐색으로 lower bound를 구한다",
    "test_inputs": ["5 7\n1 3 5 7 9", "6 4\n1 2 3 5 8 13", "4 10\n2 4 6 8"],
    "reference_code": "print(1)",
}


def make_problem(**overrides) -> GeneratedBattleProblem:
    return GeneratedBattleProblem(**{**VALID_PROBLEM, **overrides})


def make_state(**overrides) -> BattleState:
    base = {
        "category": Category.BINARY_SEARCH,
        "problem_title": VALID_PROBLEM["problem_title"],
        "problem_content": VALID_PROBLEM["problem_content"],
        "algorithm_core": VALID_PROBLEM["algorithm_core"],
        "reference_code": VALID_PROBLEM["reference_code"],
        "test_cases": [BattleTestCase(input=i) for i in VALID_PROBLEM["test_inputs"]],
    }
    return BattleState(**{**base, **overrides})


def fix_structured_response(
    monkeypatch: pytest.MonkeyPatch,
    result: GeneratedBattleProblem | None = None,
    error: Exception | None = None,
) -> list[str]:
    """구조화 호출 대신 정해진 결과를 돌려주도록 바꿔 끼운다."""
    prompts: list[str] = []

    async def fake(prompt: str, cfg, schema):
        prompts.append(prompt)
        if error:
            raise error
        return result

    monkeypatch.setattr(module, "call_llm_structured", fake)
    return prompts


def fix_run_code(monkeypatch: pytest.MonkeyPatch, results: list[RunResult]) -> None:
    """judge0 실행 대신 정해진 결과를 순서대로 돌려준다."""
    queue = iter(results)

    async def fake(code, language, stdin, limit) -> RunResult:
        return next(queue)

    monkeypatch.setattr(module, "run_code", fake)


def succeeded(stdout: str) -> RunResult:
    return RunResult(status=ExecutionStatus.SUCCEEDED, stdout=stdout)


def failed() -> RunResult:
    return RunResult(status=ExecutionStatus.RUNTIME_ERROR, stderr="boom")


# ── 문제 생성 ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_생성_결과를_상태_필드로_옮긴다(monkeypatch):
    fix_structured_response(monkeypatch, result=make_problem())
    result = await generate_battle_problem(BattleState())

    assert result["problem_title"] == VALID_PROBLEM["problem_title"]
    assert result["algorithm_core"].startswith("정렬된 배열")
    assert len(result["test_cases"]) == TEST_CASE_COUNT
    assert all(case.output == "" for case in result["test_cases"])
    assert "generate_battle_problem" in result["node_models"]


@pytest.mark.asyncio
async def test_재시도면_카테고리를_유지한다(monkeypatch):
    fix_structured_response(monkeypatch, result=make_problem())
    result = await generate_battle_problem(make_state(category=Category.GRAPH))

    assert result["category"] == Category.GRAPH


@pytest.mark.asyncio
async def test_생성에_실패하면_폐기한다(monkeypatch):
    from src.core.exception import LLMOutputParseError

    fix_structured_response(monkeypatch, error=LLMOutputParseError("형식 불일치"))
    result = await generate_battle_problem(BattleState())

    assert result["is_discarded"] is True
    assert result["discard_reason"] == DiscardReason.LLM_ERROR


def test_프롬프트에_출력을_적지_말라고_한다():
    prompt = build_prompt(Category.MATH)

    assert "출력은 적지 않는다" in prompt
    assert "MATH" in prompt


# ── 정적 검증 ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_정상이면_통과한다():
    assert await static_validate(make_state()) == {}


@pytest.mark.asyncio
async def test_제목이_비면_폐기한다():
    result = await static_validate(make_state(problem_title=""))

    assert result["discard_reason"] == DiscardReason.EMPTY_FIELD


@pytest.mark.asyncio
async def test_제목이_길면_폐기한다():
    result = await static_validate(
        make_state(problem_title="가" * (MAX_TITLE_LENGTH + 1))
    )

    assert result["discard_reason"] == DiscardReason.VALIDATION_FAILED


@pytest.mark.asyncio
async def test_지문이_길면_폐기한다():
    long_content = "가" * (MAX_CONTENT_LENGTH + 1)
    result = await static_validate(make_state(problem_content=long_content))

    assert result["discard_reason"] == DiscardReason.VALIDATION_FAILED


@pytest.mark.asyncio
async def test_테스트_입력이_모자라면_폐기한다():
    two = [BattleTestCase(input="1"), BattleTestCase(input="2")]
    result = await static_validate(make_state(test_cases=two))

    assert result["discard_reason"] == DiscardReason.VALIDATION_FAILED


@pytest.mark.asyncio
async def test_테스트_입력이_겹치면_폐기한다():
    same = [BattleTestCase(input="1") for _ in range(TEST_CASE_COUNT)]
    result = await static_validate(make_state(test_cases=same))

    assert result["discard_reason"] == DiscardReason.VALIDATION_FAILED


# ── 출력 채우기 ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_실행_결과로_출력을_채운다(monkeypatch):
    fix_run_code(monkeypatch, [succeeded("4"), succeeded("4"), succeeded("-1")])
    result = await fill_outputs(make_state())

    assert [case.output for case in result["test_cases"]] == ["4", "4", "-1"]


@pytest.mark.asyncio
async def test_실행에_실패하면_재시도_횟수를_올린다(monkeypatch):
    fix_run_code(monkeypatch, [succeeded("4"), failed(), succeeded("-1")])
    result = await fill_outputs(make_state())

    assert result == {"fill_attempt": 1}


@pytest.mark.asyncio
async def test_출력이_비면_재시도_횟수를_올린다(monkeypatch):
    fix_run_code(monkeypatch, [succeeded("4"), succeeded("  "), succeeded("-1")])
    result = await fill_outputs(make_state())

    assert result == {"fill_attempt": 1}


@pytest.mark.asyncio
async def test_재시도_한도를_넘으면_폐기한다():
    result = await fill_outputs(make_state(fill_attempt=3, max_fill_attempt=3))

    assert result["discard_reason"] == DiscardReason.MAX_ATTEMPT_EXCEEDED
