import pytest

from src.problem.nodes import generate_ref_code as module
from src.problem.nodes import stubs
from src.problem.nodes.generate_ref_code import ReferenceCode, generate_ref_code
from src.problem.state import GraphState, Language
from src.shared.llm import LLMOutputParseError


async def make_state() -> GraphState:
    """정적 검증을 통과한 문제가 담긴 상태를 만든다."""
    base = GraphState(requested_difficulty="LV2", requested_category="DP")
    return base.model_copy(update=await stubs.generate_problem(base))


def fix_structured_response(monkeypatch: pytest.MonkeyPatch, code: str) -> list[str]:
    """
    구조화 호출 대신 정해진 코드를 돌려주도록 바꿔 끼운다.

    Returns:
        list[str]: 호출 시 전달된 프롬프트가 쌓이는 목록
    """
    prompts: list[str] = []

    async def fake(prompt: str, cfg, schema):
        prompts.append(prompt)
        return ReferenceCode(code=code)

    monkeypatch.setattr(module, "call_llm_structured", fake)
    return prompts


@pytest.mark.asyncio
async def test_응답의_코드를_레퍼런스_코드로_저장한다(monkeypatch):
    fix_structured_response(monkeypatch, "print(2)")
    result = await generate_ref_code(await make_state())

    assert result["reference_code"] == "print(2)"
    assert result["reference_language"] == Language.PYTHON
    assert "generate_ref_code" in result["node_models"]


@pytest.mark.asyncio
async def test_코드_안의_코드펜스를_벗긴다(monkeypatch):
    fix_structured_response(monkeypatch, "```python\nprint(2)\n```")
    result = await generate_ref_code(await make_state())

    assert result["reference_code"] == "print(2)"


@pytest.mark.asyncio
async def test_코드가_비어_있으면_파싱_에러를_낸다(monkeypatch):
    fix_structured_response(monkeypatch, "   ")

    with pytest.raises(LLMOutputParseError):
        await generate_ref_code(await make_state())


@pytest.mark.asyncio
async def test_프롬프트에_카테고리를_넣지_않는다(monkeypatch):
    prompts = fix_structured_response(monkeypatch, "print(2)")
    state = await make_state()
    await generate_ref_code(state)

    assert state.problem_title in prompts[0]
    assert state.category_select_reason not in prompts[0]
    assert "카테고리:" not in prompts[0]
