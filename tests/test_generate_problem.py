import json

import pytest

from src.problem.nodes import generate_problem as module
from src.problem.nodes.generate_problem import build_prompt, generate_problem
from src.problem.schema import Difficulty
from src.problem.state import ConstraintDataType, GraphState, Language
from src.shared.llm import LLMOutputParseError

VALID_RESPONSE = {
    "difficulty": "LV2",
    "category": "DP",
    "category_select_reason": "부분 문제의 최적해를 누적해 구하는 문제",
    "algorithm_core": "직전 두 계단까지의 방법 수를 더해 N번째 계단의 방법 수를 누적한다",
    "problem_title": "계단 오르기",
    "problem_description": "한 번에 1칸 또는 2칸씩 오를 때 N칸 계단을 오르는 방법의 수",
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
            "min_value": "1",
            "max_value": "45",
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


def fix_llm_response(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    """Gemini 호출 대신 정해진 응답을 돌려주도록 바꿔 끼운다."""

    async def fake(prompt: str, cfg) -> str:
        return text

    monkeypatch.setattr(module, "call_llm", fake)


def make_state(category: str = "DP") -> GraphState:
    return GraphState(requested_difficulty="LV2", requested_category=category)


@pytest.mark.asyncio
async def test_정상_응답을_상태_필드로_변환한다(monkeypatch):
    fix_llm_response(monkeypatch, json.dumps(VALID_RESPONSE, ensure_ascii=False))
    result = await generate_problem(make_state())

    assert result["difficulty"] == Difficulty.LV2
    assert result["problem_title"] == "계단 오르기"
    assert result["algorithm_core"].startswith("직전 두 계단")
    assert result["input_constraints"][1].data_type == ConstraintDataType.LONG
    assert len(result["execution_limits"]) == 4
    assert "generate_problem" in result["node_models"]


@pytest.mark.asyncio
async def test_필드가_빠지면_파싱_에러를_낸다(monkeypatch):
    broken = {k: v for k, v in VALID_RESPONSE.items() if k != "problem_title"}
    fix_llm_response(monkeypatch, json.dumps(broken, ensure_ascii=False))

    with pytest.raises(LLMOutputParseError):
        await generate_problem(make_state())


@pytest.mark.asyncio
async def test_핵심_풀이_아이디어가_빠지면_파싱_에러를_낸다(monkeypatch):
    broken = {k: v for k, v in VALID_RESPONSE.items() if k != "algorithm_core"}
    fix_llm_response(monkeypatch, json.dumps(broken, ensure_ascii=False))

    with pytest.raises(LLMOutputParseError):
        await generate_problem(make_state())


@pytest.mark.asyncio
async def test_자료형이_enum에_없으면_파싱_에러를_낸다(monkeypatch):
    broken = json.loads(json.dumps(VALID_RESPONSE))
    broken["input_constraints"][0]["data_type"] = "integer"
    fix_llm_response(monkeypatch, json.dumps(broken, ensure_ascii=False))

    with pytest.raises(LLMOutputParseError):
        await generate_problem(make_state())


def test_RANDOM이면_카테고리_목록에서_고르게_한다():
    prompt = build_prompt(make_state("RANDOM"), [])

    assert "다음 중 하나를 골라" in prompt
    assert "ARRAY" in prompt
    assert "RANDOM" not in prompt


def test_카테고리를_지정하면_그대로_쓰게_한다():
    prompt = build_prompt(make_state("DP"), [])

    assert "DP (category에 그대로 적는다)" in prompt


def test_프롬프트에_핵심_풀이_아이디어를_요구한다():
    prompt = build_prompt(make_state("DP"), [])

    assert '"algorithm_core"' in prompt
