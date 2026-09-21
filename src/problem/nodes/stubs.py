"""
아직 구현되지 않은 노드의 임시 구현.

실제 노드가 완성되면 graph.py의 DEFAULT_NODES에서 해당 항목을 교체한다.
LLM을 호출하는 노드의 임시 구현은 그래프 테스트에서 계속 사용한다.
"""

from src.problem.state import (
    ConstraintDataType,
    ConstraintScope,
    Example,
    ExecutionLimit,
    GraphState,
    InputConstraint,
    Language,
    LLMConfig,
)


async def generate_problem(state: GraphState) -> dict:
    """정적 검증을 통과하는 고정 문제를 반환한다."""
    return {
        "difficulty": state.requested_difficulty,
        "category": state.requested_category,
        "category_select_reason": "임시 문제",
        "algorithm_core": "해시맵으로 M - x의 등장 횟수를 누적해 합이 M인 쌍의 수를 센다",
        "problem_title": "합이 M인 쌍의 개수",
        "problem_description": "두 원소의 합이 M이 되는 쌍의 개수를 구하시오.",
        "input_format": "첫째 줄에 N과 M, 둘째 줄에 N개의 정수가 주어진다.",
        "output_format": "쌍의 개수를 출력한다.",
        "problem_examples": [Example(input="5 6\n1 2 3 4 5", output="2")],
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


async def check_duplicate(state: GraphState) -> dict:
    """항상 신규로 판정한다."""
    return {"is_duplicated": False}


async def generate_ref_code(state: GraphState) -> dict:
    """고정 레퍼런스 코드를 반환한다."""
    return {"reference_code": "print(2)", "reference_language": Language.PYTHON}


async def semantic_validate(state: GraphState) -> dict:
    """항상 통과로 판정한다."""
    return {"is_semantically_valid": True}


async def generate_testcases(state: GraphState) -> dict:
    return {"node_models": {"generate_testcases": LLMConfig(prompt_version="stub")}}


async def generate_solution_code(state: GraphState) -> dict:
    return {"node_models": {"generate_solution_code": LLMConfig(prompt_version="stub")}}


async def generate_nl_solution(state: GraphState) -> dict:
    return {"node_models": {"generate_nl_solution": LLMConfig(prompt_version="stub")}}


async def finalize(state: GraphState) -> dict:
    """확정 후 색인·버퍼 저장 자리. 현재는 아무것도 하지 않는다."""
    return {}


async def discard_problem(state: GraphState) -> dict:
    """폐기 로그 저장 자리. 현재는 아무것도 하지 않는다."""
    return {}
