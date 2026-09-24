from collections.abc import Callable

import pytest

from src.core.enums import DiscardReason, ExecutionStatus, Language
from src.problem.nodes import generate_testcase as module
from src.problem.nodes.generate_testcase import (
    MAX_REPORTED_FAILURES,
    REQUESTED_COUNT,
    HiddenTestCaseInputs,
    build_prompt,
    generate_testcase,
    normalize_inputs,
    summarize_failures,
)
from src.problem.state import (
    HIDDEN_TEST_CASE_COUNT,
    GraphState,
    ProblemExample,
)
from src.shared.code_runner import RunResult
from tests import fake_nodes

# 첫 줄의 두 수를 더해 출력하는, 실제로 도는 레퍼런스 코드
REFERENCE_CODE = "print(sum(map(int, input().split())))"


def many_inputs(count: int = HIDDEN_TEST_CASE_COUNT) -> list[str]:
    """서로 다른 입력 count개."""
    return [f"{index} {index + 1}\n{index}" for index in range(count)]


async def make_state(**updates) -> GraphState:
    """의미 검증까지 통과한 상태를 만든다."""
    base = GraphState(requested_difficulty="LV2", requested_category="HASH_TABLE")
    state = base.model_copy(update=await fake_nodes.generate_problem(base))
    return state.model_copy(update={"reference_code": REFERENCE_CODE, **updates})


def succeeded(stdout: str) -> RunResult:
    return RunResult(status=ExecutionStatus.SUCCEEDED, stdout=stdout, exit_code=0)


def fix_run_code(
    monkeypatch: pytest.MonkeyPatch, make_result: Callable[[str], RunResult]
) -> list[str]:
    """
    레퍼런스 실행을 가짜로 바꿔 끼운다.

    Returns:
        list[str]: 실행에 넘어간 표준 입력이 쌓이는 목록
    """
    seen: list[str] = []

    async def fake(code: str, language: Language, stdin: str, limit) -> RunResult:
        seen.append(stdin)
        return make_result(stdin)

    monkeypatch.setattr(module, "run_code", fake)
    return seen


def fix_structured_response(
    monkeypatch: pytest.MonkeyPatch, inputs: list[str]
) -> list[str]:
    """
    구조화 호출 대신 정해진 입력 목록을 돌려준다.

    Returns:
        list[str]: 호출 시 전달된 프롬프트가 쌓이는 목록
    """
    prompts: list[str] = []

    async def fake(prompt: str, cfg, schema):
        prompts.append(prompt)
        return HiddenTestCaseInputs(inputs=inputs)

    monkeypatch.setattr(module, "call_llm_structured", fake)
    return prompts


# ── 프롬프트 ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_prompt_carries_the_input_format_and_constraints() -> None:
    state = await make_state()

    prompt = build_prompt(state)

    assert state.input_format in prompt
    assert state.input_constraints[0].target in prompt


@pytest.mark.asyncio
async def test_prompt_asks_for_more_than_it_keeps() -> None:
    """중복과 예시 겹침으로 줄어드는 몫이 있어 여유분을 요청한다."""
    assert REQUESTED_COUNT > HIDDEN_TEST_CASE_COUNT
    assert str(REQUESTED_COUNT) in build_prompt(await make_state())


@pytest.mark.asyncio
async def test_prompt_hides_the_category() -> None:
    """입력을 만드는 데는 입력 형식과 제약만 필요하다."""
    state = await make_state()

    assert state.category_select_reason not in build_prompt(state)


# ── 정리 ───────────────────────────────────────────────────────────────


def test_normalize_strips_and_drops_blanks() -> None:
    given = ["  1 2 ", "", "   ", "3 4"]

    assert normalize_inputs(given, []) == ["1 2", "3 4"]


def test_normalize_removes_duplicates() -> None:
    assert normalize_inputs(["1 2", "1 2", " 1 2 ", "3 4"], []) == ["1 2", "3 4"]


def test_normalize_drops_inputs_already_shown_as_examples() -> None:
    """공개 예시와 같은 입력은 비공개 테스트 케이스로서 값이 없다."""
    examples = [ProblemExample(input="1 2", output="3")]

    assert normalize_inputs(["1 2", "3 4"], examples) == ["3 4"]


def test_normalize_caps_at_the_required_count() -> None:
    given = many_inputs(HIDDEN_TEST_CASE_COUNT + 5)

    assert len(normalize_inputs(given, [])) == HIDDEN_TEST_CASE_COUNT


# ── 노드 ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_node_fills_output_from_the_reference_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """기대 출력은 모델에게 묻지 않고 레퍼런스 코드를 돌려서 얻는다."""
    fix_structured_response(monkeypatch, many_inputs())
    fix_run_code(monkeypatch, lambda stdin: succeeded(f"답: {stdin.splitlines()[0]}\n"))

    result = await generate_testcase(await make_state())
    cases = result["hidden_test_cases"]

    assert len(cases) == HIDDEN_TEST_CASE_COUNT
    assert cases[0].output == "답: 0 1"  # 끝의 개행은 떼고 담는다
    assert "is_discarded" not in result
    assert result["node_models"]["generate_testcase"].prompt_version == (
        module.PROMPT_VERSION
    )


@pytest.mark.asyncio
async def test_node_runs_the_reference_once_per_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fix_structured_response(monkeypatch, many_inputs())
    seen = fix_run_code(monkeypatch, lambda stdin: succeeded("1"))

    await generate_testcase(await make_state())

    assert seen == many_inputs()


@pytest.mark.asyncio
async def test_node_runs_the_real_reference_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """실행기를 바꿔 끼우지 않고 끝까지 돌려 본다."""
    fix_structured_response(monkeypatch, many_inputs())

    result = await generate_testcase(await make_state())
    cases = result["hidden_test_cases"]

    assert len(cases) == HIDDEN_TEST_CASE_COUNT
    # 입력 "0 1" -> 0 + 1 = 1, "1 2" -> 3, "2 3" -> 5 ...
    assert [case.output for case in cases[:3]] == ["1", "3", "5"]


@pytest.mark.asyncio
async def test_node_discards_when_the_reference_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """한 입력이라도 실패하면 레퍼런스 코드나 제약이 잘못된 것이다."""
    fix_structured_response(monkeypatch, many_inputs())
    inputs = many_inputs()

    def result_for(stdin: str) -> RunResult:
        if stdin == inputs[5]:
            return RunResult(status=ExecutionStatus.TIMED_OUT)
        return succeeded("1")

    fix_run_code(monkeypatch, result_for)

    result = await generate_testcase(await make_state())

    assert result["is_discarded"] is True
    assert result["discard_reason"] == DiscardReason.REFERENCE_CODE_FAILED
    assert "입력 6" in result["discard_detail"]
    assert ExecutionStatus.TIMED_OUT.value in result["discard_detail"]


@pytest.mark.asyncio
async def test_node_discards_when_the_reference_prints_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """출력이 없는 테스트 케이스는 채점에 쓸 수 없다."""
    fix_structured_response(monkeypatch, many_inputs())
    fix_run_code(monkeypatch, lambda stdin: succeeded("   \n"))

    result = await generate_testcase(await make_state())

    assert result["is_discarded"] is True
    assert result["discard_reason"] == DiscardReason.REFERENCE_CODE_FAILED
    assert "출력이 비었음" in result["discard_detail"]


@pytest.mark.asyncio
async def test_node_reports_only_a_few_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """모두 실패해도 폐기 사유에 전부 적지 않는다."""
    fix_structured_response(monkeypatch, many_inputs())
    fix_run_code(monkeypatch, lambda stdin: RunResult(status=ExecutionStatus.TIMED_OUT))

    result = await generate_testcase(await make_state())
    remaining = HIDDEN_TEST_CASE_COUNT - MAX_REPORTED_FAILURES

    assert f"외 {remaining}건" in result["discard_detail"]


def test_summarize_failures_keeps_the_message_short() -> None:
    assert summarize_failures(["a", "b"]) == "a; b"
    assert summarize_failures(["a", "b", "c", "d", "e"]) == "a; b; c 외 2건"


@pytest.mark.asyncio
async def test_node_discards_without_a_reference_code() -> None:
    result = await generate_testcase(await make_state(reference_code=None))

    assert result["is_discarded"] is True
    assert result["discard_reason"] == DiscardReason.EMPTY_FIELD


@pytest.mark.asyncio
async def test_node_discards_without_a_python_execution_limit() -> None:
    state = await make_state()
    without_python = [
        limit
        for limit in state.execution_limits
        if limit.language is not Language.PYTHON
    ]

    result = await generate_testcase(
        state.model_copy(update={"execution_limits": without_python})
    )

    assert result["is_discarded"] is True
    assert result["discard_reason"] == DiscardReason.EMPTY_FIELD


@pytest.mark.asyncio
async def test_margin_absorbs_a_collision_with_a_public_example(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """공개 예시와 겹친 입력 하나 때문에 문제가 날아가면 안 된다.

    공개 예시는 프롬프트에 그대로 들어가므로 모델이 거기서 값을 따올 수 있다.
    """
    state = await make_state()
    collided = [state.problem_examples[0].input, *many_inputs(REQUESTED_COUNT - 1)]
    fix_structured_response(monkeypatch, collided)
    fix_run_code(monkeypatch, lambda stdin: succeeded("1"))

    result = await generate_testcase(state)

    assert "is_discarded" not in result
    assert len(result["hidden_test_cases"]) == HIDDEN_TEST_CASE_COUNT


@pytest.mark.asyncio
async def test_node_discards_when_too_few_cases_survive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fix_structured_response(monkeypatch, many_inputs(HIDDEN_TEST_CASE_COUNT - 1))

    result = await generate_testcase(await make_state())

    assert result["is_discarded"] is True
    assert result["discard_reason"] == DiscardReason.LLM_ERROR
    assert f"{HIDDEN_TEST_CASE_COUNT - 1}개" in result["discard_detail"]


@pytest.mark.asyncio
async def test_duplicates_count_against_the_required_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """중복을 걷어낸 뒤의 개수로 판단한다."""
    fix_structured_response(monkeypatch, ["1 2"] * HIDDEN_TEST_CASE_COUNT)

    result = await generate_testcase(await make_state())

    assert result["is_discarded"] is True


@pytest.mark.asyncio
async def test_node_discards_on_any_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """키 누락, 타임아웃, 레이트 리밋도 폐기로 바뀌어야 한다."""

    async def fake(prompt: str, cfg, schema):
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")

    monkeypatch.setattr(module, "call_llm_structured", fake)

    result = await generate_testcase(await make_state())

    assert result["is_discarded"] is True
    assert result["discard_reason"] == DiscardReason.LLM_ERROR


@pytest.mark.asyncio
async def test_node_survives_a_runner_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """실행기가 예외를 올려도 그래프를 죽이지 않는다."""
    fix_structured_response(monkeypatch, many_inputs())

    async def broken(code, language, stdin, limit):
        raise OSError("프로세스를 띄울 수 없음")

    monkeypatch.setattr(module, "run_code", broken)

    result = await generate_testcase(await make_state())

    assert result["is_discarded"] is True
    assert result["discard_reason"] == DiscardReason.REFERENCE_CODE_FAILED
