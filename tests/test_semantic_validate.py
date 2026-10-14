import pytest

from src.client.judge0 import RunResult
from src.core.enums import DiscardReason, ExecutionStatus, Language
from src.problem.nodes import semantic_validate as module
from src.problem.nodes.semantic_validate import (
    check_examples,
    normalize_output,
    semantic_validate,
)
from src.problem.state import GraphState, ProblemExample, find_limit, normalize_io_text
from tests import fake_nodes

# 첫 줄의 두 수를 더하는 레퍼런스 코드. 예시 "5 6\n1 2 3 4 5" -> 11
REFERENCE_CODE = "print(sum(map(int, input().split())))"


async def make_state(**updates) -> GraphState:
    """레퍼런스 코드까지 만들어진 상태. 공개 예시의 답은 11이다."""
    base = GraphState(requested_difficulty="LV2", requested_category="HASH")
    state = base.model_copy(update=await fake_nodes.generate_problem(base))
    return state.model_copy(
        update={
            "reference_code": REFERENCE_CODE,
            "problem_examples": [ProblemExample(input="5 6\n1 2 3 4 5", output="11")],
            **updates,
        }
    )


def fix_run(monkeypatch: pytest.MonkeyPatch, result: RunResult) -> list[str]:
    """레퍼런스 실행 결과를 고정한다."""
    seen: list[str] = []

    async def fake(code, language, stdin, limit):
        seen.append(stdin)
        return result

    monkeypatch.setattr(module, "run_code", fake)
    return seen


def succeeded(stdout: str) -> RunResult:
    return RunResult(status=ExecutionStatus.SUCCEEDED, stdout=stdout)


# ── 입출력 줄바꿈 정규화 ──────────────────────────────────────────────


def test_escaped_newline_is_restored() -> None:
    assert normalize_io_text("5\\n4 1 2") == "5\n4 1 2"


def test_real_newline_is_kept() -> None:
    assert normalize_io_text("5\n4 1 2") == "5\n4 1 2"


def test_backslash_without_newline_escape_is_kept() -> None:
    assert normalize_io_text("a\\b") == "a\\b"


def test_problem_example_normalizes_on_parse() -> None:
    example = ProblemExample(input="2\\n3 4", output="7")
    assert example.input == "2\n3 4"


# ── 출력 비교 ──────────────────────────────────────────────────────────


def test_trailing_newline_is_ignored() -> None:
    assert normalize_output("11\n") == normalize_output("11")


def test_trailing_spaces_per_line_are_ignored() -> None:
    assert normalize_output("1 2  \n3 4\t") == normalize_output("1 2\n3 4")


def test_trailing_blank_lines_are_ignored() -> None:
    assert normalize_output("11\n\n\n") == normalize_output("11")


def test_leading_whitespace_is_kept() -> None:
    """오른쪽 정렬 출력처럼 앞 공백이 의미 있는 경우가 있다."""
    assert normalize_output("  *\n ***") != normalize_output("*\n***")


def test_different_values_stay_different() -> None:
    assert normalize_output("11") != normalize_output("12")


# ── 실행 기반 검사 ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_escaped_newline_reaches_judge0_as_real_newline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = await make_state(
        problem_examples=[ProblemExample(input="5 6\\n1 2 3 4 5", output="11")]
    )
    seen = fix_run(monkeypatch, succeeded("11"))

    limit = find_limit(state.execution_limits, Language.PYTHON)
    assert await check_examples(state, limit) is None
    assert seen == ["5 6\n1 2 3 4 5"]


@pytest.mark.asyncio
async def test_matching_examples_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    fix_run(monkeypatch, succeeded("11\n"))

    result = await semantic_validate(await make_state())

    assert result["is_semantically_valid"] is True
    assert "is_discarded" not in result
    assert result["semantic_validation_attempt"] == 1


@pytest.mark.asyncio
async def test_real_reference_code_reproduces_the_example(
    monkeypatch: pytest.MonkeyPatch, judge0_available: None
) -> None:
    """실행기를 바꿔 끼우지 않고 judge0까지 끝까지 돌려 본다."""

    result = await semantic_validate(await make_state())

    assert result["is_semantically_valid"] is True


@pytest.mark.asyncio
async def test_every_example_is_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    runs = fix_run(monkeypatch, succeeded("11"))
    state = await make_state(
        problem_examples=[
            ProblemExample(input="5 6", output="11"),
            ProblemExample(input="1 2", output="11"),
        ]
    )

    await semantic_validate(state)

    assert runs == ["5 6", "1 2"]


@pytest.mark.asyncio
async def test_mismatch_retries_while_attempts_remain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """코드가 틀렸을 수 있으니 다시 만들어 본다. 폐기가 아니다."""
    fix_run(monkeypatch, succeeded("99"))

    result = await semantic_validate(await make_state())

    assert result["is_semantically_valid"] is False
    assert "is_discarded" not in result
    assert result["semantic_validation_attempt"] == 1


@pytest.mark.asyncio
async def test_mismatch_on_the_last_attempt_is_discarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fix_run(monkeypatch, succeeded("99"))
    state = await make_state(semantic_validation_attempt=2)  # max_attempt=3

    result = await semantic_validate(state)

    assert result["is_discarded"] is True
    assert result["discard_reason"] == DiscardReason.EXAMPLE_MISMATCH
    assert "3회 시도" in result["discard_detail"]


@pytest.mark.asyncio
async def test_execution_failure_reports_reference_code_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fix_run(monkeypatch, RunResult(status=ExecutionStatus.TIMED_OUT))
    state = await make_state(semantic_validation_attempt=2)

    result = await semantic_validate(state)

    assert result["discard_reason"] == DiscardReason.REFERENCE_CODE_FAILED
    assert ExecutionStatus.TIMED_OUT.value in result["discard_detail"]


@pytest.mark.asyncio
async def test_executor_failure_is_discarded_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """채점 서버 장애는 코드를 다시 만들어도 풀리지 않으니 첫 시도에서 바로 폐기한다."""
    fix_run(
        monkeypatch,
        RunResult(status=ExecutionStatus.INTERNAL_ERROR, stderr="ReadTimeout('')"),
    )

    result = await semantic_validate(await make_state())

    assert result["is_discarded"] is True
    assert result["discard_reason"] == DiscardReason.EXECUTOR_ERROR
    assert "ReadTimeout" in result["discard_detail"]
    assert result["semantic_validation_attempt"] == 1


@pytest.mark.asyncio
async def test_execution_failure_still_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """코드가 실제로 실패한 경우는 지금처럼 다시 만들어 본다."""
    fix_run(monkeypatch, RunResult(status=ExecutionStatus.TIMED_OUT))

    result = await semantic_validate(await make_state())

    assert result["is_semantically_valid"] is False
    assert "is_discarded" not in result


@pytest.mark.asyncio
async def test_check_examples_returns_none_when_all_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fix_run(monkeypatch, succeeded("11"))
    state = await make_state()
    limit = find_limit(state.execution_limits, Language.PYTHON)

    assert await check_examples(state, limit) is None


# ── 사전 조건 ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_reference_code_is_discarded() -> None:
    result = await semantic_validate(await make_state(reference_code=None))

    assert result["discard_reason"] == DiscardReason.EMPTY_FIELD


@pytest.mark.asyncio
async def test_missing_examples_is_discarded() -> None:
    result = await semantic_validate(await make_state(problem_examples=[]))

    assert result["discard_reason"] == DiscardReason.EMPTY_FIELD


@pytest.mark.asyncio
async def test_missing_python_execution_limit_is_discarded() -> None:
    state = await make_state()
    without_python = [
        limit
        for limit in state.execution_limits
        if limit.language is not Language.PYTHON
    ]

    result = await semantic_validate(
        state.model_copy(update={"execution_limits": without_python})
    )

    assert result["discard_reason"] == DiscardReason.EMPTY_FIELD
