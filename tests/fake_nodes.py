"""그래프 노드의 테스트 대역.

프로덕션 코드는 이 모듈을 쓰지 않는다. 그래프를 돌리는 테스트가 바깥을
건드리지 않게 하고(LLM 호출, DB 쓰기), 검증을 통과한 상태를 만드는 데 쓴다.

노드를 실 구현으로 바꿀 때마다 OFFLINE_NODES에 한 줄을 추가한다.
빠뜨리면 테스트가 실제 API를 때리거나 개발용 DB에 쓰레기 행을 쌓는다.
"""

from src.client.llm import LLMConfig
from src.core.enums import ConstraintDataType, ConstraintScope, Language
from src.problem.state import (
    ExecutionLimit,
    GraphState,
    InputConstraint,
    ProblemExample,
)


async def generate_problem(state: GraphState) -> dict:
    """정적 검증을 통과하는 고정 문제를 반환한다."""
    return {
        "difficulty": state.requested_difficulty,
        "category": state.requested_category,
        "category_select_reason": "임시 문제",
        "algorithm_core": (
            "해시맵으로 M - x의 등장 횟수를 누적해 합이 M인 쌍의 수를 센다"
        ),
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
                min_value=2,
                max_value=100000,
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


async def generate_testcase(state: GraphState) -> dict:
    return {"node_models": {"generate_testcase": LLMConfig(prompt_version="stub")}}


async def generate_solution_code(state: GraphState) -> dict:
    return {"node_models": {"generate_solution_code": LLMConfig(prompt_version="stub")}}


async def generate_nl_keyword(state: GraphState) -> dict:
    return {"node_models": {"generate_nl_keyword": LLMConfig(prompt_version="stub")}}


async def finalize(state: GraphState) -> dict:
    """저장하지 않는다. 테스트가 개발용 DB에 쓰지 않게 막는다."""
    return {}


async def discard_problem(state: GraphState) -> dict:
    """폐기 로그를 남기지 않는다. 테스트가 개발용 DB에 쓰지 않게 막는다."""
    return {}


OFFLINE_NODES = {
    # LLM 호출
    "generate_problem": generate_problem,
    "generate_ref_code": generate_ref_code,
    "semantic_validate": semantic_validate,
    "generate_testcase": generate_testcase,
    "generate_solution_code": generate_solution_code,
    "generate_nl_keyword": generate_nl_keyword,
    # 임베딩 호출 + DB 조회
    "check_duplicate": check_duplicate,
    # DB 쓰기
    "finalize": finalize,
    "discard_problem": discard_problem,
}


def offline_except(*under_test: str, **overrides) -> dict:
    """
    검사할 노드만 실 구현으로 남기고 나머지를 임시 구현으로 고정한다.

    Parameters:
        under_test (str): 실 구현으로 돌릴 노드 이름. build_graph의 기본값을 쓴다
        overrides: 직접 지정할 노드

    Returns:
        dict: build_graph에 넘길 overrides
    """
    kept = {
        name: node for name, node in OFFLINE_NODES.items() if name not in under_test
    }
    return kept | overrides
