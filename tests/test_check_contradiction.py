import pytest

from src.core.enums import DiscardReason
from src.problem.nodes import check_contradiction as module
from src.problem.nodes.check_contradiction import (
    LogicVerdict,
    build_logic_prompt,
    check_contradiction,
)
from src.problem.state import GraphState, ProblemExample
from tests import fake_nodes


async def make_state(**updates) -> GraphState:
    """problem 필드가 채워진 상태"""
    base = GraphState(requested_difficulty="LV2", requested_category="HASH")
    state = base.model_copy(update=await fake_nodes.generate_problem(base))
    return state.model_copy(
        update={
            "problem_examples": [
                ProblemExample(
                    input="5 6\n1 2 3 4 5",
                    output="11",
                )
            ],
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


# ── 모순 검사 ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_logic_prompt_carries_problem_and_category() -> None:
    state = await make_state()

    prompt = build_logic_prompt(state)

    assert state.problem_content in prompt
    assert state.category_select_reason in prompt


@pytest.mark.asyncio
async def test_problem_contradiction_is_discarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fix_verdict(monkeypatch, "problem")

    result = await check_contradiction(await make_state())

    assert result["is_discarded"] is True
    assert result["discard_reason"] == DiscardReason.PROBLEM_CONTRADICTION
    assert result["discard_detail"] == "판단 근거"


@pytest.mark.asyncio
async def test_constraint_contradiction_is_discarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fix_verdict(monkeypatch, "constraint")

    result = await check_contradiction(await make_state())

    assert result["discard_reason"] == DiscardReason.CONSTRAINT_CONTRADICTION


@pytest.mark.asyncio
async def test_no_contradiction_is_not_discarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fix_verdict(monkeypatch, "none")

    result = await check_contradiction(await make_state())

    assert "is_discarded" not in result


@pytest.mark.asyncio
async def test_llm_failure_is_discarded_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake(prompt, cfg, schema):
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")

    monkeypatch.setattr(module, "call_llm_structured", fake)

    result = await check_contradiction(await make_state())

    assert result["discard_reason"] == DiscardReason.LLM_ERROR
