import pytest

from src.client.code_runner import RunResult
from src.core.enums import DiscardReason, ExecutionStatus, Language
from src.problem.nodes import semantic_validate as module
from src.problem.nodes.semantic_validate import (
    LogicVerdict,
    build_logic_prompt,
    check_examples,
    normalize_output,
    semantic_validate,
)
from src.problem.state import GraphState, ProblemExample, find_limit
from tests import fake_nodes
from tests.fake_nodes import offline_except

# 첫 줄의 두 수를 더하는 레퍼런스 코드. 예시 "5 6\n1 2 3 4 5" -> 11
REFERENCE_CODE = "print(sum(map(int, input().split())))"


async def make_state(**updates) -> GraphState:
    """레퍼런스 코드까지 만들어진 상태. 공개 예시의 답은 11이다."""
    base = GraphState(requested_difficulty="LV2", requested_category="HASH_TABLE")
    state = base.model_copy(update=await fake_nodes.generate_problem(base))
    return state.model_copy(
        update={
            "reference_code": REFERENCE_CODE,
            "problem_examples": [ProblemExample(input="5 6\n1 2 3 4 5", output="11")],
            **updates,
        }
    )


def fix_verdict(monkeypatch: pytest.MonkeyPatch, contradiction: str) -> list[str]:
    """모순 검사 응답을 고정한다.

    Returns:
        list[str]: 호출 시 전달된 프롬프트가 쌓이는 목록
    """
    prompts: list[str] = []

    async def fake(prompt: str, cfg, schema):
        prompts.append(prompt)
        return LogicVerdict(contradiction=contradiction, reason="판단 근거")

    monkeypatch.setattr(module, "call_llm_structured", fake)
    return prompts


def fix_run(monkeypatch: pytest.MonkeyPatch, result: RunResult) -> list[str]:
    """레퍼런스 실행 결과를 고정한다."""
    seen: list[str] = []

    async def fake(code, language, stdin, limit):
        seen.append(stdin)
        return result

    monkeypatch.setattr(module, "run_code", fake)
    return seen


def succeeded(stdout: str) -> RunResult:
    return RunResult(status=ExecutionStatus.SUCCEEDED, stdout=stdout, exit_code=0)


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


# ── 모순 검사 ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_logic_prompt_carries_problem_and_category() -> None:
    state = await make_state()

    prompt = build_logic_prompt(state)

    assert state.problem_content in prompt
    assert state.category_select_reason in prompt  # 검수자는 분류를 안다


@pytest.mark.asyncio
async def test_problem_contradiction_is_discarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fix_verdict(monkeypatch, "problem")
    runs = fix_run(monkeypatch, succeeded("11"))

    result = await semantic_validate(await make_state())

    assert result["is_discarded"] is True
    assert result["discard_reason"] == DiscardReason.PROBLEM_CONTRADICTION
    assert result["discard_detail"] == "판단 근거"
    assert runs == []  # 모순이면 실행하지 않는다


@pytest.mark.asyncio
async def test_constraint_contradiction_is_discarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fix_verdict(monkeypatch, "constraint")
    fix_run(monkeypatch, succeeded("11"))

    result = await semantic_validate(await make_state())

    assert result["discard_reason"] == DiscardReason.CONSTRAINT_CONTRADICTION


@pytest.mark.asyncio
async def test_contradiction_check_runs_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """실행이 비싸므로 모순 검사로 먼저 걸러 낸다."""
    order: list[str] = []

    async def fake_llm(prompt, cfg, schema):
        order.append("llm")
        return LogicVerdict(contradiction="none", reason="")

    async def fake_run(code, language, stdin, limit):
        order.append("run")
        return succeeded("11")

    monkeypatch.setattr(module, "call_llm_structured", fake_llm)
    monkeypatch.setattr(module, "run_code", fake_run)

    await semantic_validate(await make_state())

    assert order == ["llm", "run"]


@pytest.mark.asyncio
async def test_contradiction_check_is_skipped_on_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """지문은 코드를 다시 만들어도 그대로다. 두 번 물어볼 이유가 없다."""
    prompts = fix_verdict(monkeypatch, "none")
    fix_run(monkeypatch, succeeded("11"))

    await semantic_validate(await make_state(semantic_validation_attempt=1))

    assert prompts == []


@pytest.mark.asyncio
async def test_llm_failure_is_discarded_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake(prompt, cfg, schema):
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")

    monkeypatch.setattr(module, "call_llm_structured", fake)

    result = await semantic_validate(await make_state())

    assert result["discard_reason"] == DiscardReason.LLM_ERROR


# ── 실행 기반 검사 ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_matching_examples_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    fix_verdict(monkeypatch, "none")
    fix_run(monkeypatch, succeeded("11\n"))

    result = await semantic_validate(await make_state())

    assert result["is_semantically_valid"] is True
    assert "is_discarded" not in result
    assert result["semantic_validation_attempt"] == 1


@pytest.mark.asyncio
async def test_real_reference_code_reproduces_the_example(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """실행기를 바꿔 끼우지 않고 끝까지 돌려 본다."""
    fix_verdict(monkeypatch, "none")

    result = await semantic_validate(await make_state())

    assert result["is_semantically_valid"] is True


@pytest.mark.asyncio
async def test_every_example_is_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    fix_verdict(monkeypatch, "none")
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
    fix_verdict(monkeypatch, "none")
    fix_run(monkeypatch, succeeded("99"))

    result = await semantic_validate(await make_state())

    assert result["is_semantically_valid"] is False
    assert "is_discarded" not in result
    assert result["semantic_validation_attempt"] == 1


@pytest.mark.asyncio
async def test_mismatch_on_the_last_attempt_is_discarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fix_verdict(monkeypatch, "none")
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
    fix_verdict(monkeypatch, "none")
    fix_run(monkeypatch, RunResult(status=ExecutionStatus.TIMED_OUT))
    state = await make_state(semantic_validation_attempt=2)

    result = await semantic_validate(state)

    assert result["discard_reason"] == DiscardReason.REFERENCE_CODE_FAILED
    assert ExecutionStatus.TIMED_OUT.value in result["discard_detail"]


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


# ── 그래프 안에서 ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_contradiction_check_costs_one_call_across_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """재시도 루프가 모순 검사를 반복해서는 안 된다.

    실행 기반 검사가 계속 실패하면 그래프가 generate_ref_code로 되돌아오는데,
    그때마다 모순 검사를 다시 부르면 LLM 호출이 시도 횟수만큼 늘어난다.
    """
    from src.problem.graph import build_graph

    calls = {"logic": 0, "ref_code": 0}

    async def fake_logic(prompt, cfg, schema):
        calls["logic"] += 1
        return LogicVerdict(contradiction="none", reason="")

    async def fake_ref_code(state: GraphState) -> dict:
        calls["ref_code"] += 1
        return {"reference_code": REFERENCE_CODE, "reference_language": "python"}

    monkeypatch.setattr(module, "call_llm_structured", fake_logic)
    fix_run(monkeypatch, succeeded("맞지 않는 출력"))

    graph = build_graph(
        overrides=offline_except("semantic_validate", generate_ref_code=fake_ref_code)
    )
    state = GraphState(
        **await graph.ainvoke(
            GraphState(requested_difficulty="LV2", requested_category="HASH_TABLE")
        )
    )

    assert calls["logic"] == 1
    assert calls["ref_code"] == state.semantic_validation_max_attempt
    assert state.discard_reason == DiscardReason.EXAMPLE_MISMATCH


@pytest.mark.asyncio
async def test_contradiction_skips_the_retry_loop_entirely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """모순이면 코드를 다시 만들지도, 실행하지도 않는다."""
    from src.problem.graph import build_graph

    calls = {"ref_code": 0, "run": 0}

    async def fake_ref_code(state: GraphState) -> dict:
        calls["ref_code"] += 1
        return {"reference_code": REFERENCE_CODE, "reference_language": "python"}

    async def counting_run(code, language, stdin, limit):
        calls["run"] += 1
        return succeeded("11")

    fix_verdict(monkeypatch, "constraint")
    monkeypatch.setattr(module, "run_code", counting_run)

    graph = build_graph(
        overrides=offline_except("semantic_validate", generate_ref_code=fake_ref_code)
    )
    state = GraphState(
        **await graph.ainvoke(
            GraphState(requested_difficulty="LV2", requested_category="HASH_TABLE")
        )
    )

    assert calls["ref_code"] == 1  # 되돌아오지 않는다
    assert calls["run"] == 0  # 실행은 아예 하지 않는다
    assert state.discard_reason == DiscardReason.CONSTRAINT_CONTRADICTION
