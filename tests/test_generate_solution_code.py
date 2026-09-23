import pytest

from src.enum import DiscardReason, Language
from src.problem.nodes import generate_solution_code as module
from src.problem.nodes.generate_ref_code import REFERENCE_LANGUAGE
from src.problem.nodes.generate_solution_code import (
    LanguageSolution,
    SolutionHint,
    build_hint_prompt,
    build_translate_prompt,
    generate_solution_code,
    solve_language,
)
from src.problem.state import GraphState, LLMConfig
from src.shared.llm import LLMOutputParseError
from tests import fake_nodes

REFERENCE_CODE = "import sys\nprint(2)"


async def make_state(**updates) -> GraphState:
    """의미 검증까지 통과한 상태를 만든다."""
    base = GraphState(requested_difficulty="LV2", requested_category="HASH_TABLE")
    state = base.model_copy(update=await fake_nodes.generate_problem(base))
    return state.model_copy(
        update={
            "reference_code": REFERENCE_CODE,
            "reference_language": REFERENCE_LANGUAGE,
            **updates,
        }
    )


def fix_responses(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """언어별 호출에 스키마에 맞는 응답을 돌려준다.

    Returns:
        list[str]: 호출 시 전달된 프롬프트가 쌓이는 목록
    """
    prompts: list[str] = []

    async def fake(prompt: str, cfg, schema):
        prompts.append(prompt)
        if schema is SolutionHint:
            return SolutionHint(hint="파이썬 힌트")
        return LanguageSolution(code="translated code", hint="번역 힌트")

    monkeypatch.setattr(module, "call_llm_structured", fake)
    return prompts


def fix_failure_for(monkeypatch: pytest.MonkeyPatch, language: Language) -> None:
    """지정한 언어의 호출만 실패시킨다."""

    async def fake(prompt: str, cfg, schema):
        if language.value in prompt:
            raise LLMOutputParseError(f"{language.value} 실패")
        if schema is SolutionHint:
            return SolutionHint(hint="파이썬 힌트")
        return LanguageSolution(code="translated code", hint="번역 힌트")

    monkeypatch.setattr(module, "call_llm_structured", fake)


# ── 프롬프트 ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_translate_prompt_gives_the_verified_python_solution() -> None:
    state = await make_state()
    limit = next(
        limit for limit in state.execution_limits if limit.language is Language.JAVA
    )

    prompt = build_translate_prompt(state, Language.JAVA, limit)

    assert REFERENCE_CODE in prompt
    assert Language.JAVA.value in prompt
    assert str(limit.time_limit_ms) in prompt


@pytest.mark.asyncio
async def test_translate_prompt_hides_the_category() -> None:
    """풀이자는 지문만으로 풀 수 있어야 한다."""
    state = await make_state()
    limit = state.execution_limits[0]

    prompt = build_translate_prompt(state, Language.JAVA, limit)

    assert state.category_select_reason not in prompt


@pytest.mark.asyncio
async def test_hint_prompt_asks_only_for_a_hint() -> None:
    prompt = build_hint_prompt(await make_state())

    assert REFERENCE_CODE in prompt
    assert "code에는" not in prompt


# ── 언어 하나 ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reference_language_reuses_the_verified_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """검증을 통과한 코드를 버리고 다시 만들지 않는다."""
    fix_responses(monkeypatch)
    state = await make_state()
    limit = state.execution_limits[0]

    code, hint = await solve_language(state, REFERENCE_LANGUAGE, limit, LLMConfig())

    assert code == REFERENCE_CODE
    assert hint == "파이썬 힌트"


@pytest.mark.asyncio
async def test_other_languages_are_translated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fix_responses(monkeypatch)
    state = await make_state()
    limit = state.execution_limits[0]

    code, hint = await solve_language(state, Language.CPP, limit, LLMConfig())

    assert code == "translated code"
    assert hint == "번역 힌트"


@pytest.mark.asyncio
async def test_code_fence_is_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake(prompt: str, cfg, schema):
        return LanguageSolution(code="```java\nclass Main {}\n```", hint="힌트")

    monkeypatch.setattr(module, "call_llm_structured", fake)
    state = await make_state()

    code, _ = await solve_language(
        state, Language.JAVA, state.execution_limits[0], LLMConfig()
    )

    assert code == "class Main {}"


@pytest.mark.asyncio
async def test_empty_code_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake(prompt: str, cfg, schema):
        return LanguageSolution(code="   ", hint="힌트")

    monkeypatch.setattr(module, "call_llm_structured", fake)
    state = await make_state()

    with pytest.raises(LLMOutputParseError):
        await solve_language(
            state, Language.JAVA, state.execution_limits[0], LLMConfig()
        )


# ── 노드 ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_node_fills_every_language(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = fix_responses(monkeypatch)

    result = await generate_solution_code(await make_state())

    assert len(prompts) == len(Language)  # 네 언어를 동시에 호출한다
    assert [code.language for code in result["solution_codes"]] == list(Language)
    assert [comment.language for comment in result["hint_comments"]] == list(Language)
    assert "is_discarded" not in result


@pytest.mark.asyncio
async def test_node_pairs_each_code_with_its_own_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fix_responses(monkeypatch)

    result = await generate_solution_code(await make_state())
    codes = {code.language: code.content for code in result["solution_codes"]}
    hints = {comment.language: comment.content for comment in result["hint_comments"]}

    assert codes[REFERENCE_LANGUAGE] == REFERENCE_CODE
    assert hints[REFERENCE_LANGUAGE] == "파이썬 힌트"
    assert codes[Language.JAVA] == "translated code"
    assert hints[Language.JAVA] == "번역 힌트"


@pytest.mark.asyncio
async def test_node_records_its_model_config(monkeypatch: pytest.MonkeyPatch) -> None:
    fix_responses(monkeypatch)

    result = await generate_solution_code(await make_state())

    assert result["node_models"]["generate_solution_code"].prompt_version == (
        module.PROMPT_VERSION
    )


@pytest.mark.asyncio
async def test_node_discards_when_one_language_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """실행 제한이 네 언어에 다 있으니 정답 코드도 다 채워야 한다."""
    fix_failure_for(monkeypatch, Language.CPP)

    result = await generate_solution_code(await make_state())

    assert result["is_discarded"] is True
    assert result["discard_reason"] == DiscardReason.LLM_ERROR
    assert Language.CPP.value in result["discard_detail"]


@pytest.mark.asyncio
async def test_node_discards_on_any_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """키 누락, 타임아웃, 레이트 리밋도 폐기로 바뀌어야 한다."""

    async def fake(prompt: str, cfg, schema):
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")

    monkeypatch.setattr(module, "call_llm_structured", fake)

    result = await generate_solution_code(await make_state())

    assert result["is_discarded"] is True
    assert result["discard_reason"] == DiscardReason.LLM_ERROR


@pytest.mark.asyncio
async def test_node_discards_without_a_reference_code() -> None:
    result = await generate_solution_code(await make_state(reference_code=None))

    assert result["is_discarded"] is True
    assert result["discard_reason"] == DiscardReason.EMPTY_FIELD


@pytest.mark.asyncio
async def test_node_discards_when_a_language_has_no_execution_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fix_responses(monkeypatch)
    state = await make_state()
    without_cpp = [
        limit for limit in state.execution_limits if limit.language is not Language.CPP
    ]

    result = await generate_solution_code(
        state.model_copy(update={"execution_limits": without_cpp})
    )

    assert result["is_discarded"] is True
    assert result["discard_reason"] == DiscardReason.EMPTY_FIELD
    assert Language.CPP.value in result["discard_detail"]
