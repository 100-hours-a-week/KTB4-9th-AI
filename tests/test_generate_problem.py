import contextlib
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.core.enums import Category, ConstraintDataType, Difficulty, Language
from src.core.exception import LLMOutputParseError
from src.problem.nodes import generate_problem as module
from src.problem.nodes.generate_problem import (
    RESPONSE_CATEGORIES,
    GeneratedProblem,
    build_prompt,
    fetch_few_shot,
    generate_problem,
    resolve_category,
)
from src.problem.state import GraphState, ProblemExample

VALID_PROBLEM = {
    "difficulty": "LV2",
    "category": "DP",
    "category_select_reason": "부분 문제의 최적해를 누적해 구하는 문제",
    "algorithm_core": (
        "직전 두 계단까지의 방법 수를 더해 N번째 계단의 방법 수를 누적한다"
    ),
    "problem_title": "계단 오르기",
    "problem_content": "한 번에 1칸 또는 2칸씩 오를 때 N칸 계단을 오르는 방법의 수",
    "input_format": "첫째 줄에 N이 주어진다.",
    "output_format": "방법의 수를 출력한다.",
    "problem_examples": [
        {"input": "3", "output": "3", "explanation": "1+1+1, 1+2, 2+1"}
    ],
    "input_constraints": [
        {
            "target": "N",
            "scope": "input",
            "data_type": "int",
            "min_value": 1,
            "max_value": 45,
            "data_count": 1,
            "special_conditions": [],
        },
        {"target": "output", "scope": "output", "data_type": "long"},
    ],
    "execution_limits": [
        {"language": lang.value, "time_limit_ms": 2000, "memory_limit_kb": 262144}
        for lang in Language
    ],
}


def make_problem(**overrides) -> GeneratedProblem:
    return GeneratedProblem(**{**VALID_PROBLEM, **overrides})


def fix_structured_response(
    monkeypatch: pytest.MonkeyPatch,
    result: GeneratedProblem | None = None,
    error: Exception | None = None,
) -> None:
    """구조화 호출 대신 정해진 객체나 에러를 돌려주도록 바꿔 끼운다."""

    async def fake(prompt: str, cfg, schema):
        if error:
            raise error
        return result

    monkeypatch.setattr(module, "call_llm_structured", fake)
    fix_few_shot(monkeypatch, [])


def fix_few_shot(monkeypatch: pytest.MonkeyPatch, problems: list[dict]) -> dict:
    """참고 문제 조회 대신 정해진 목록을 돌려주도록 바꿔 끼운다.

    Returns:
        dict: 조회 인자가 담긴다
    """
    seen: dict = {}

    async def fake(difficulty, category, k):
        seen.update(difficulty=difficulty, category=category, k=k)
        return problems

    monkeypatch.setattr(module, "fetch_few_shot", fake)
    return seen


def fix_seed_lookup(monkeypatch: pytest.MonkeyPatch, seeds: list) -> dict:
    """DB 세션과 시드 저장소를 가짜로 바꿔 끼운다.

    Returns:
        dict: 조회 인자가 담긴다
    """
    seen: dict = {}

    @contextlib.asynccontextmanager
    async def fake_scope():
        yield object()

    class FakeRepository:
        def __init__(self, session) -> None:
            pass

        async def sample(self, category, difficulty, k):
            seen.update(category=category, difficulty=difficulty, k=k)
            return seeds

    monkeypatch.setattr(module, "session_scope", fake_scope)
    monkeypatch.setattr(module, "FewshotSeedRepository", FakeRepository)
    return seen


def make_seed() -> SimpleNamespace:
    return SimpleNamespace(
        problem_title="피보나치 수",
        problem_content="N번째 피보나치 수를 구한다.",
        input_format="첫째 줄에 N이 주어진다.",
        output_format="N번째 피보나치 수를 출력한다.",
    )


def make_state(category: str = "DP") -> GraphState:
    return GraphState(requested_difficulty="LV2", requested_category=category)


@pytest.mark.asyncio
async def test_구조화_응답을_상태_필드로_옮긴다(monkeypatch):
    fix_structured_response(monkeypatch, result=make_problem())
    result = await generate_problem(make_state())

    assert result["difficulty"] == Difficulty.LV2
    assert result["category"] == "DP"
    assert result["problem_title"] == "계단 오르기"
    assert result["algorithm_core"].startswith("직전 두 계단")
    assert isinstance(result["problem_examples"][0], ProblemExample)
    assert result["input_constraints"][1].data_type == ConstraintDataType.LONG
    assert len(result["execution_limits"]) == 4
    assert "generate_problem" in result["node_models"]


@pytest.mark.asyncio
async def test_RANDOM이면_코드가_고른_카테고리로_조회하고_생성한다(monkeypatch):
    prompts: list[str] = []

    async def fake(prompt: str, cfg, schema):
        prompts.append(prompt)
        return make_problem(category="GRAPH")

    monkeypatch.setattr(module, "call_llm_structured", fake)
    monkeypatch.setattr(module.random, "choice", lambda _: Category.GRAPH)
    seen = fix_few_shot(monkeypatch, [])

    result = await generate_problem(make_state("RANDOM"))

    assert seen["category"] == Category.GRAPH
    assert "GRAPH (category에 그대로 적는다)" in prompts[0]
    assert result["category"] == "GRAPH"


@pytest.mark.asyncio
async def test_응답_카테고리가_다르면_코드가_정한_카테고리를_쓴다(monkeypatch):
    fix_structured_response(monkeypatch, result=make_problem(category="GREEDY"))

    result = await generate_problem(make_state("DP"))

    assert result["category"] == "DP"


@pytest.mark.asyncio
async def test_구조화_호출이_실패하면_파싱_에러가_전달된다(monkeypatch):
    fix_structured_response(monkeypatch, error=LLMOutputParseError("형식 불일치"))

    with pytest.raises(LLMOutputParseError):
        await generate_problem(make_state())


def test_응답_형식은_목록에_없는_카테고리를_거부한다():
    with pytest.raises(ValidationError):
        make_problem(category="동적 계획법")


def test_응답_형식은_핵심_풀이_아이디어가_빠지면_거부한다():
    data = {k: v for k, v in VALID_PROBLEM.items() if k != "algorithm_core"}

    with pytest.raises(ValidationError):
        GeneratedProblem(**data)


def test_RANDOM이면_응답_카테고리_중_하나를_고른다():
    picked = {resolve_category("RANDOM") for _ in range(200)}

    assert Category.RANDOM not in picked
    assert picked <= set(RESPONSE_CATEGORIES)
    assert len(picked) > 1


def test_카테고리를_지정하면_그대로_쓴다():
    assert resolve_category("DP") == Category.DP


def test_카테고리를_그대로_쓰게_한다():
    prompt = build_prompt(Difficulty.LV2, Category.DP, [])

    assert "DP (category에 그대로 적는다)" in prompt
    assert "RANDOM" not in prompt


def test_프롬프트에_핵심_풀이_아이디어_규칙이_있다():
    prompt = build_prompt(Difficulty.LV2, Category.DP, [])

    assert "algorithm_core" in prompt


@pytest.mark.asyncio
async def test_참고_문제를_프롬프트에_넣는다(monkeypatch):
    prompts: list[str] = []

    async def fake(prompt: str, cfg, schema):
        prompts.append(prompt)
        return make_problem()

    monkeypatch.setattr(module, "call_llm_structured", fake)
    fix_few_shot(monkeypatch, [{"problem_title": "피보나치 수"}])

    await generate_problem(make_state())

    assert "제목: 피보나치 수" in prompts[0]


@pytest.mark.asyncio
async def test_시드를_난이도_카테고리로_조회해_지문_필드만_넘긴다(monkeypatch):
    seen = fix_seed_lookup(monkeypatch, [make_seed()])

    problems = await fetch_few_shot(Difficulty.LV2, Category.DP, 3)

    assert seen == {"category": Category.DP, "difficulty": Difficulty.LV2, "k": 3}
    assert problems == [
        {
            "problem_title": "피보나치 수",
            "problem_content": "N번째 피보나치 수를 구한다.",
            "input_format": "첫째 줄에 N이 주어진다.",
            "output_format": "N번째 피보나치 수를 출력한다.",
        }
    ]


@pytest.mark.asyncio
async def test_RANDOM이면_시드를_조회하지_않는다(monkeypatch):
    seen = fix_seed_lookup(monkeypatch, [make_seed()])

    assert await fetch_few_shot(Difficulty.LV2, Category.RANDOM, 3) == []
    assert seen == {}


@pytest.mark.asyncio
async def test_시드_조회가_실패하면_빈_목록으로_진행한다(monkeypatch):
    @contextlib.asynccontextmanager
    async def broken_scope():
        raise OSError("connection refused")
        yield  # pragma: no cover

    monkeypatch.setattr(module, "session_scope", broken_scope)

    assert await fetch_few_shot(Difficulty.LV2, Category.DP, 3) == []
