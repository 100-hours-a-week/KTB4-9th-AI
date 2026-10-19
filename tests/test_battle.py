import pytest

from src.client import llm
from src.client.judge0 import RunResult
from src.client.llm import current_local_model
from src.core.config import get_settings
from src.core.enums import Category, DiscardReason, ExecutionStatus, Trigger
from src.problem.battle import nodes as module
from src.problem.battle.nodes import (
    ANSWER_GUIDE,
    GeneratedBattleProblem,
    build_prompt,
    ensure_answer_guide,
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
) -> list:
    """구조화 호출 대신 정해진 결과를 돌려주도록 바꿔 끼운다.

    재시도·폴백 헬퍼는 진짜로 돌게 두고, 그 아래의 call_llm_structured만 바꾼다.

    Returns:
        list: 호출될 때마다 그 순간의 로컬 모델(없으면 None)이 담긴다
    """
    seen: list = []

    async def fake(prompt: str, cfg, schema):
        seen.append(current_local_model())
        if error:
            raise error
        return result

    monkeypatch.setattr(llm, "call_llm_structured", fake)
    return seen


def set_batch_model(monkeypatch: pytest.MonkeyPatch, model: str | None) -> None:
    """배치가 부를 vLLM 모델 이름을 테스트 값으로 바꾼다. 폴백은 끈다."""
    settings = get_settings()
    monkeypatch.setattr(settings, "batch_local_model", model)
    monkeypatch.setattr(settings, "batch_local_thinking", None)
    monkeypatch.setattr(settings, "batch_local_fallback", False)
    monkeypatch.setattr(llm, "LOCAL_RETRY_DELAYS_S", (0, 0))


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


@pytest.mark.asyncio
async def test_배치면_vLLM으로_만들고_그_모델을_기록한다(monkeypatch):
    set_batch_model(monkeypatch, "qwen-batch")
    seen = fix_structured_response(monkeypatch, result=make_problem())

    result = await generate_battle_problem(BattleState(trigger=Trigger.BATCH))

    assert [m.model_name for m in seen] == ["qwen-batch"], (
        "배치인데 vLLM으로 가지 않았다"
    )
    recorded = result["node_models"]["generate_battle_problem"]
    assert recorded.model_name == "qwen-batch"
    assert recorded.temperature is None, "Gemini의 temperature가 기록에 남았다"


@pytest.mark.asyncio
async def test_온디맨드면_배치_모델이_있어도_Gemini로_만든다(monkeypatch):
    set_batch_model(monkeypatch, "qwen-batch")
    seen = fix_structured_response(monkeypatch, result=make_problem())

    result = await generate_battle_problem(BattleState(trigger=Trigger.ON_DEMAND))

    assert seen == [None], "온디맨드가 vLLM으로 갔다"
    recorded = result["node_models"]["generate_battle_problem"]
    assert recorded.model_name == get_settings().llm_models.generate_battle_problem


@pytest.mark.asyncio
async def test_배치에서_로컬_모델이_형식을_못_맞추면_폐기한다(monkeypatch):
    from src.core.exception import LLMOutputParseError

    set_batch_model(monkeypatch, "qwen-batch")
    seen = fix_structured_response(
        monkeypatch, error=LLMOutputParseError("형식 불일치")
    )

    result = await generate_battle_problem(BattleState(trigger=Trigger.BATCH))

    assert len(seen) == 3, "형식 오류를 재시도하지 않았다"
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


def test_안내_문구가_없으면_지문_끝에_붙인다():
    content = "두 수의 합을 구하시오."
    result = ensure_answer_guide(content)
    assert result.endswith(ANSWER_GUIDE)
    assert content in result


def test_안내_문구가_이미_있으면_다시_붙이지_않는다():
    content = f"두 수의 합을 구하시오.\n\n{ANSWER_GUIDE}"
    result = ensure_answer_guide(content)
    assert result.count(ANSWER_GUIDE) == 1


def test_프롬프트가_코드_제출_표현을_막고_안내_문구를_요구한다():
    prompt = build_prompt(Category.BINARY_SEARCH)
    assert "프로그램을 작성하세요" in prompt
    assert ANSWER_GUIDE in prompt
    assert "코드를 제출하지 않는다" in prompt
