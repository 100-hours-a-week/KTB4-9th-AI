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
            "problem_examples": [
                ProblemExample(
                    input="5 6\n1 2 3 4 5", description="계산 과정", output="11"
                )
            ],
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
    example = ProblemExample(input="2\\n3 4", description="계산 과정", output="7")
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
        problem_examples=[
            ProblemExample(
                input="5 6\\n1 2 3 4 5", description="계산 과정", output="11"
            )
        ]
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
            ProblemExample(input="5 6", description="계산 과정", output="11"),
            ProblemExample(input="1 2", description="계산 과정", output="11"),
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


# ── 같은 실패 반복 시 조기 중단 ──────────────────────────────────────────


def fix_run_by_input(
    monkeypatch: pytest.MonkeyPatch, results: dict[str, RunResult]
) -> None:
    """예시 입력마다 다른 실행 결과를 돌려준다."""

    async def fake(code, language, stdin, limit):
        return results[stdin]

    monkeypatch.setattr(module, "run_code", fake)


def next_attempt(state: GraphState, result: dict) -> GraphState:
    """노드 결과를 반영해 다음 시도의 상태를 만든다. 그래프가 하는 일과 같다."""
    return state.model_copy(update=result)


@pytest.mark.asyncio
async def test_first_failure_is_recorded_for_next_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fix_run(monkeypatch, succeeded("99"))

    result = await semantic_validate(await make_state())

    assert "is_discarded" not in result
    assert result["last_semantic_failure"]


@pytest.mark.asyncio
async def test_same_mismatch_twice_is_discarded_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """코드를 다시 만들어도 같은 오답이면 문제 결함으로 보고 남은 횟수를 쓰지 않는다."""
    fix_run(monkeypatch, succeeded("99"))
    state = await make_state()
    first = await semantic_validate(state)

    fix_run(monkeypatch, succeeded("99  \n"))  # 정규화하면 같은 출력
    result = await semantic_validate(next_attempt(state, first))

    assert result["is_discarded"] is True
    assert result["discard_reason"] == DiscardReason.EXAMPLE_MISMATCH
    assert "같은 실패 반복" in result["discard_detail"]
    assert result["semantic_validation_attempt"] == 2  # max_attempt=3보다 먼저


@pytest.mark.asyncio
async def test_different_mismatch_keeps_retrying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fix_run(monkeypatch, succeeded("99"))
    state = await make_state()
    first = await semantic_validate(state)

    fix_run(monkeypatch, succeeded("98"))
    result = await semantic_validate(next_attempt(state, first))

    assert "is_discarded" not in result
    assert result["last_semantic_failure"] != first["last_semantic_failure"]


@pytest.mark.asyncio
async def test_same_runtime_status_twice_is_discarded_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """stderr 내용이 달라도 같은 예시에서 같은 실행 상태면 같은 실패로 본다."""
    fix_run(
        monkeypatch,
        RunResult(status=ExecutionStatus.RUNTIME_ERROR, stderr="line 3\nIndexError"),
    )
    state = await make_state()
    first = await semantic_validate(state)

    fix_run(
        monkeypatch,
        RunResult(status=ExecutionStatus.RUNTIME_ERROR, stderr="line 7\nKeyError"),
    )
    result = await semantic_validate(next_attempt(state, first))

    assert result["is_discarded"] is True
    assert result["discard_reason"] == DiscardReason.REFERENCE_CODE_FAILED
    assert "같은 실패 반복" in result["discard_detail"]


@pytest.mark.asyncio
async def test_same_status_on_another_example_keeps_retrying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timed_out = RunResult(status=ExecutionStatus.TIMED_OUT)
    state = await make_state(
        problem_examples=[
            ProblemExample(input="5 6", description="계산 과정", output="11"),
            ProblemExample(input="1 2", description="계산 과정", output="3"),
        ]
    )
    fix_run_by_input(monkeypatch, {"5 6": timed_out, "1 2": timed_out})
    first = await semantic_validate(state)

    fix_run_by_input(monkeypatch, {"5 6": succeeded("11"), "1 2": timed_out})
    result = await semantic_validate(next_attempt(state, first))

    assert "is_discarded" not in result
    assert result["is_semantically_valid"] is False


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
