import pytest

from src.core.enums import DiscardReason
from src.problem.nodes import generate_nl_keyword as module
from src.problem.nodes.generate_nl_keyword import (
    KEYWORD_MAX,
    KEYWORD_MIN,
    SolutionKeywords,
    build_prompt,
    generate_nl_keyword,
    normalize_keywords,
)
from src.problem.state import GraphState
from src.shared.llm import LLMOutputParseError
from tests import fake_nodes


async def make_state() -> GraphState:
    """정적 검증을 통과한 문제가 담긴 상태를 만든다."""
    base = GraphState(requested_difficulty="LV2", requested_category="HASH_TABLE")
    return base.model_copy(update=await fake_nodes.generate_problem(base))


def fix_structured_response(
    monkeypatch: pytest.MonkeyPatch, keywords: list[str]
) -> list[str]:
    """
    구조화 호출 대신 정해진 키워드를 돌려주도록 바꿔 끼운다.

    Returns:
        list[str]: 호출 시 전달된 프롬프트가 쌓이는 목록
    """
    prompts: list[str] = []

    async def fake(prompt: str, cfg, schema):
        prompts.append(prompt)
        return SolutionKeywords(keywords=keywords)

    monkeypatch.setattr(module, "call_llm_structured", fake)
    return prompts


def fix_failure(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    async def fake(prompt: str, cfg, schema):
        raise error

    monkeypatch.setattr(module, "call_llm_structured", fake)


# ── 프롬프트 ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_prompt_carries_problem_and_algorithm_core() -> None:
    state = await make_state()

    prompt = build_prompt(state)

    assert state.problem_title in prompt
    assert state.problem_content in prompt
    assert state.algorithm_core in prompt


@pytest.mark.asyncio
async def test_prompt_includes_category_as_a_hint() -> None:
    """채점자는 문제 분류를 아는 상태이고, 분류가 기대 키워드의 단서가 된다."""
    state = await make_state()

    prompt = build_prompt(state)

    assert state.category in prompt
    assert state.category_select_reason in prompt


@pytest.mark.asyncio
async def test_prompt_states_the_keyword_range() -> None:
    prompt = build_prompt(await make_state())

    assert f"{KEYWORD_MIN}~{KEYWORD_MAX}" in prompt


@pytest.mark.asyncio
async def test_prompt_survives_missing_algorithm_core() -> None:
    state = (await make_state()).model_copy(update={"algorithm_core": None})

    assert "(없음)" in build_prompt(state)


# ── 정리 ───────────────────────────────────────────────────────────────


def test_normalize_strips_and_drops_blanks() -> None:
    given = ["  해시맵 ", "", "   ", "누적합"]

    assert normalize_keywords(given) == ["해시맵", "누적합"]


def test_normalize_removes_duplicates_ignoring_case() -> None:
    assert normalize_keywords(["HashMap", "해시맵", "hashmap"]) == [
        "HashMap",
        "해시맵",
    ]


def test_normalize_keeps_the_model_order() -> None:
    assert normalize_keywords(["둘", "하나", "셋"]) == ["둘", "하나", "셋"]


def test_normalize_caps_at_the_maximum() -> None:
    many = [f"키워드{index}" for index in range(KEYWORD_MAX + 5)]

    assert len(normalize_keywords(many)) == KEYWORD_MAX


# ── 노드 ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_node_returns_normalized_keywords_and_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fix_structured_response(monkeypatch, [" 해시맵 ", "해시맵", "투 포인터", "누적합"])

    result = await generate_nl_keyword(await make_state())

    assert result["solution_keywords"] == ["해시맵", "투 포인터", "누적합"]
    assert result["node_models"]["generate_nl_keyword"].prompt_version == (
        module.PROMPT_VERSION
    )
    assert "is_discarded" not in result


@pytest.mark.asyncio
async def test_node_discards_when_the_model_call_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """병렬 노드는 예외를 올리지 않는다. 올리면 그래프 전체가 죽는다."""
    fix_failure(monkeypatch, LLMOutputParseError("구조화 응답이 비어 있음"))

    result = await generate_nl_keyword(await make_state())

    assert result["is_discarded"] is True
    assert result["discard_reason"] == DiscardReason.LLM_ERROR


@pytest.mark.asyncio
async def test_node_discards_on_any_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """키 누락, 타임아웃, 레이트 리밋도 폐기로 바뀌어야 한다.

    LLMOutputParseError만 잡으면 이런 예외가 그래프 전체를 죽인다.
    """
    fix_failure(monkeypatch, RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다."))

    result = await generate_nl_keyword(await make_state())

    assert result["is_discarded"] is True
    assert result["discard_reason"] == DiscardReason.LLM_ERROR


@pytest.mark.asyncio
async def test_node_discards_when_too_few_keywords_survive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fix_structured_response(monkeypatch, ["해시맵", " 해시맵 ", ""])

    result = await generate_nl_keyword(await make_state())

    assert result["is_discarded"] is True
    assert result["discard_reason"] == DiscardReason.LLM_ERROR
    assert "1개" in result["discard_detail"]


@pytest.mark.asyncio
async def test_node_discards_on_an_empty_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fix_structured_response(monkeypatch, [])

    result = await generate_nl_keyword(await make_state())

    assert result["is_discarded"] is True
