import pytest

from src.enum import (
    ConstraintDataType,
    ConstraintScope,
    Difficulty,
    DiscardReason,
    Language,
)
from src.problem.nodes.static_validate import static_validate
from src.problem.state import (
    ExecutionLimit,
    GraphState,
    InputConstraint,
    ProblemExample,
)


def make_state(**overrides) -> GraphState:
    """정적 검증을 통과하는 기본 상태를 만들고, 필요한 필드만 바꿔 끼운다."""
    base = {
        "requested_difficulty": Difficulty.LV2,
        "requested_category": "HASH_TABLE",
        "difficulty": Difficulty.LV2,
        "category": "HASH_TABLE",
        "category_select_reason": "M - x 존재 여부를 O(1)에 조회하는 것이 핵심",
        "problem_title": "합이 M인 쌍의 개수",
        "problem_content": "두 원소의 합이 M이 되는 쌍의 개수를 구하시오.",
        "input_format": "첫째 줄에 N과 M, 둘째 줄에 N개의 정수가 주어진다.",
        "output_format": "쌍의 개수를 출력한다.",
        "problem_examples": [ProblemExample(input="5 6\n1 2 3 4 5", output="2")],
        "input_constraints": [
            InputConstraint(
                target="N",
                scope=ConstraintScope.INPUT,
                data_type=ConstraintDataType.INT,
                min_value="2",
                max_value="10^5",
                data_count=1,
            ),
            InputConstraint(
                target="output",
                scope=ConstraintScope.OUTPUT,
                data_type=ConstraintDataType.LONG,
            ),
        ],
        "execution_limits": [
            ExecutionLimit(language=lang, time_limit_ms=2000, memory_limit_kb=262144)
            for lang in Language
        ],
    }
    base.update(overrides)
    return GraphState(**base)


@pytest.mark.asyncio
async def test_정상_문제는_통과한다():
    result = await static_validate(make_state())
    assert result == {"is_statically_validated": True}


@pytest.mark.asyncio
async def test_제목이_비면_폐기한다():
    result = await static_validate(make_state(problem_title=""))
    assert result["discard_reason"] == DiscardReason.EMPTY_FIELD


@pytest.mark.asyncio
async def test_실행제한_언어가_빠지면_폐기한다():
    only_python = [
        ExecutionLimit(
            language=Language.PYTHON, time_limit_ms=2000, memory_limit_kb=262144
        )
    ]
    result = await static_validate(make_state(execution_limits=only_python))
    assert result["discard_reason"] == DiscardReason.EMPTY_FIELD


@pytest.mark.asyncio
async def test_난이도가_다르면_폐기한다():
    result = await static_validate(make_state(difficulty=Difficulty.LV4))
    assert result["discard_reason"] == DiscardReason.MISMATCH_LABEL


@pytest.mark.asyncio
async def test_정수_출력에_글자가_오면_폐기한다():
    examples = [ProblemExample(input="5 6\n1 2 3 4 5", output="YES")]
    result = await static_validate(make_state(problem_examples=examples))
    assert result["discard_reason"] == DiscardReason.MISMATCH_TYPE


@pytest.mark.asyncio
async def test_최솟값이_최댓값보다_크면_폐기한다():
    constraints = [
        InputConstraint(
            target="N",
            scope=ConstraintScope.INPUT,
            data_type=ConstraintDataType.INT,
            min_value="100",
            max_value="10",
        )
    ]
    result = await static_validate(make_state(input_constraints=constraints))
    assert result["discard_reason"] == DiscardReason.CONSTRAINT_CONFLICT


@pytest.mark.asyncio
async def test_서로_다른_값을_범위보다_많이_요구하면_폐기한다():
    constraints = [
        InputConstraint(
            target="arr",
            scope=ConstraintScope.INPUT,
            data_type=ConstraintDataType.INT,
            min_value="1",
            max_value="10",
            data_count=20,
            special_conditions=["서로 다른 수"],
        )
    ]
    result = await static_validate(make_state(input_constraints=constraints))
    assert result["discard_reason"] == DiscardReason.CONSTRAINT_CONFLICT
