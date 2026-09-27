from collections.abc import Awaitable, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.problem.battle.nodes import (
    check_duplicate,
    fill_outputs,
    finalize,
    generate_battle_problem,
    static_validate,
)
from src.problem.battle.state import BattleState

NodeFn = Callable[[BattleState], Awaitable[dict]]


async def discard_problem(state: BattleState) -> dict:
    """
    폐기 사유를 로그로 남긴다.

    일반 문제와 달리 배틀은 폐기 로그를 따로 저장하지 않는다.

    Parameters:
        state (BattleState): 폐기로 끝난 상태

    Returns:
        dict: 상태를 바꾸지 않으므로 빈 dict
    """
    return {}


DEFAULT_NODES: dict[str, NodeFn] = {
    "generate_battle_problem": generate_battle_problem,
    "static_validate": static_validate,
    "fill_outputs": fill_outputs,
    "check_duplicate": check_duplicate,
    "finalize": finalize,
    "discard_problem": discard_problem,
}


def route_after_generate(state: BattleState) -> str:
    """생성에 실패하면 폐기, 성공하면 정적 검증으로 보낸다."""
    return "discard_problem" if state.is_discarded else "static_validate"


def route_after_static(state: BattleState) -> str:
    """정적 검증에 걸리면 폐기, 통과하면 출력 채우기로 보낸다."""
    return "discard_problem" if state.is_discarded else "fill_outputs"


def route_after_fill(state: BattleState) -> str:
    """
    출력 채우기 결과에 따라 분기한다.

    폐기면 폐기, 출력이 다 채워졌으면 중복 검사,
    실행에 실패했으면 문제부터 다시 만든다.
    """
    if state.is_discarded:
        return "discard_problem"
    if all(case.output for case in state.test_cases):
        return "check_duplicate"
    return "generate_battle_problem"


def route_after_duplicate(state: BattleState) -> str:
    """중복이면 폐기, 신규면 저장으로 보낸다."""
    return "discard_problem" if state.is_discarded else "finalize"


def build_battle_graph(
    overrides: dict[str, NodeFn] | None = None,
) -> CompiledStateGraph:
    """
    배틀 문제 생성 그래프를 조립한다.

    일반 문제 생성보다 단계를 줄여 LLM 1회, 코드 실행 3회로 구성한다.

    Parameters:
        overrides (dict[str, NodeFn] | None): 기본 노드 대신 사용할 함수. 테스트용

    Returns:
        CompiledStateGraph: 실행 가능한 그래프
    """
    nodes = {**DEFAULT_NODES, **(overrides or {})}

    builder = StateGraph(BattleState)
    for name, fn in nodes.items():
        builder.add_node(name, fn)

    builder.add_edge(START, "generate_battle_problem")
    builder.add_conditional_edges(
        "generate_battle_problem",
        route_after_generate,
        ["static_validate", "discard_problem"],
    )
    builder.add_conditional_edges(
        "static_validate",
        route_after_static,
        ["fill_outputs", "discard_problem"],
    )
    builder.add_conditional_edges(
        "fill_outputs",
        route_after_fill,
        ["check_duplicate", "generate_battle_problem", "discard_problem"],
    )
    builder.add_conditional_edges(
        "check_duplicate",
        route_after_duplicate,
        ["finalize", "discard_problem"],
    )
    builder.add_edge("finalize", END)
    builder.add_edge("discard_problem", END)

    return builder.compile()


def get_battle_graph() -> CompiledStateGraph:
    """그래프는 한 번만 조립한다. 요청마다 다시 만들 이유가 없다."""
    return build_battle_graph()
