from collections.abc import Awaitable, Callable
from functools import lru_cache

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.problem.nodes.check_duplicate import check_duplicate
from src.problem.nodes.discard_problem import discard_problem
from src.problem.nodes.finalize import finalize
from src.problem.nodes.generate_nl_keyword import generate_nl_keyword
from src.problem.nodes.generate_problem import generate_problem
from src.problem.nodes.generate_ref_code import generate_ref_code
from src.problem.nodes.generate_solution_code import generate_solution_code
from src.problem.nodes.generate_testcase import generate_testcase
from src.problem.nodes.semantic_validate import semantic_validate
from src.problem.nodes.static_validate import static_validate
from src.problem.state import GraphState

NodeFn = Callable[[GraphState], Awaitable[dict]]

PARALLEL_NODES = [
    "generate_testcase",
    "generate_solution_code",
    "generate_nl_keyword",
]


def tag_discard_stage(name: str, node: NodeFn) -> NodeFn:
    """폐기 결과에 그 노드의 이름을 붙인다.

    노드마다 자기 이름을 넘기게 하면 빠뜨리기 쉽고, DiscardedProblem.stage는
    비워 둘 수 없는 컬럼이다. 배선하는 쪽에서 채우면 잊을 일이 없다.
    """

    async def tagged(state: GraphState) -> dict:
        result = await node(state)
        if result.get("is_discarded") and not result.get("discard_stage"):
            return result | {"discard_stage": name}
        return result

    return tagged


async def collect_parallel(state: GraphState) -> dict:
    """병렬 생성 노드 셋의 합류 지점. 상태를 바꾸지 않는다.

    병렬 노드에 조건부 엣지를 직접 달면 성공한 노드는 자기 결과만 반영된
    상태를 보고 finalize로 가 버린다. 배리어를 하나 두어야 셋의 결과가
    모두 병합된 상태에서 한 번만 분기할 수 있다.
    """
    return {}


# 노드 이름 → 실행 함수. 테스트는 overrides로 일부를 바꿔 끼운다.
DEFAULT_NODES: dict[str, NodeFn] = {
    "generate_problem": generate_problem,
    "static_validate": static_validate,
    "check_duplicate": check_duplicate,
    "generate_ref_code": generate_ref_code,
    "semantic_validate": semantic_validate,
    "generate_testcase": generate_testcase,
    "generate_solution_code": generate_solution_code,
    "generate_nl_keyword": generate_nl_keyword,
    "collect_parallel": collect_parallel,
    "finalize": finalize,
    "discard_problem": discard_problem,
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


def route_after_parallel(state: GraphState) -> str:
    """병렬 생성 결과 합류 후. 하나라도 폐기했으면 저장하지 않는다."""
    return "discard_problem" if state.is_discarded else "finalize"


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
        builder.add_node(name, tag_discard_stage(name, fn))

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
    builder.add_edge(PARALLEL_NODES, "collect_parallel")  # 셋 다 끝날 때까지 대기
    builder.add_conditional_edges(
        "collect_parallel", route_after_parallel, ["finalize", "discard_problem"]
    )
    builder.add_edge("finalize", END)
    builder.add_edge("discard_problem", END)

    return builder.compile()


@lru_cache(maxsize=1)
def get_graph():
    """그래프는 한 번만 조립한다. 요청마다 다시 만들 이유가 없다."""
    return build_graph()
