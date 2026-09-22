from collections.abc import Awaitable, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from src.problem.nodes.generate_ref_code import generate_ref_code

from src.problem.nodes import stubs
from src.problem.nodes.generate_problem import generate_problem
from src.problem.nodes.static_validate import static_validate
from src.problem.state import GraphState

NodeFn = Callable[[GraphState], Awaitable[dict]]

PARALLEL_NODES = [
    "generate_testcases",
    "generate_solution_code",
    "generate_nl_solution",
]

# 노드 이름 → 실행 함수. 실제 노드가 완성되면 stubs 항목을 교체한다.
DEFAULT_NODES: dict[str, NodeFn] = {
    "generate_problem": generate_problem,
    "static_validate": static_validate,
    "check_duplicate": stubs.check_duplicate,
    "generate_ref_code": generate_ref_code,
    "semantic_validate": stubs.semantic_validate,
    "generate_testcases": stubs.generate_testcases,
    "generate_solution_code": stubs.generate_solution_code,
    "generate_nl_solution": stubs.generate_nl_solution,
    "finalize": stubs.finalize,
    "discard_problem": stubs.discard_problem,
}


def route_after_static(state: GraphState) -> str:
    """정적 검증 결과에 따라 중복 검사 또는 폐기로 보낸다."""
    return "discard_problem" if state.is_discarded else "check_duplicate"


def route_after_duplicate(state: GraphState) -> str:
    """중복이면 폐기, 신규면 검증용 코드 생성으로 보낸다."""
    if state.is_discarded or state.is_duplicated:
        return "discard_problem"
    return "generate_ref_code"


def route_after_semantic(state: GraphState) -> str | list[str]:
    """
    의미 검증 결과에 따라 분기한다.

    폐기면 폐기, 통과면 후속 생성 노드 3개를 병렬 실행,
    그 외(실행 오류·예시 불일치)는 검증용 코드를 다시 생성한다.
    """
    if state.is_discarded:
        return "discard_problem"
    if state.is_semantically_valid:
        return PARALLEL_NODES
    return "generate_ref_code"


def build_graph(overrides: dict[str, NodeFn] | None = None) -> CompiledStateGraph:
    """
    문제 생성 그래프를 조립한다.

    Parameters:
        overrides (dict[str, NodeFn] | None): 기본 노드 대신 사용할 함수. 테스트용

    Returns:
        CompiledStateGraph: 실행 가능한 그래프
    """
    nodes = {**DEFAULT_NODES, **(overrides or {})}

    builder = StateGraph(GraphState)
    for name, fn in nodes.items():
        builder.add_node(name, fn)

    builder.add_edge(START, "generate_problem")
    builder.add_edge("generate_problem", "static_validate")
    builder.add_conditional_edges(
        "static_validate",
        route_after_static,
        ["check_duplicate", "discard_problem"],
    )
    builder.add_conditional_edges(
        "check_duplicate",
        route_after_duplicate,
        ["generate_ref_code", "discard_problem"],
    )
    builder.add_edge("generate_ref_code", "semantic_validate")
    builder.add_conditional_edges(
        "semantic_validate",
        route_after_semantic,
        [*PARALLEL_NODES, "generate_ref_code", "discard_problem"],
    )
    builder.add_edge(PARALLEL_NODES, "finalize")
    builder.add_edge("finalize", END)
    builder.add_edge("discard_problem", END)

    return builder.compile()
