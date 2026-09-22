from src.problem.state import GraphState, InputConstraint


def render_constraint(constraint: InputConstraint) -> str:
    """
    제약 하나를 프롬프트용 한 줄로 만든다.

    Parameters:
        constraint (InputConstraint): 입력 또는 출력 제약

    Returns:
        str: 대상, 범위, 개수, 추가 조건을 담은 한 줄
    """
    line = f"- {constraint.target} ({constraint.scope}, {constraint.data_type})"
    if constraint.min_value is not None or constraint.max_value is not None:
        line += f": {constraint.min_value} ~ {constraint.max_value}"
    if constraint.data_count is not None:
        line += f", 개수 {constraint.data_count}"
    if constraint.special_conditions:
        line += f", 조건: {', '.join(constraint.special_conditions)}"
    return line


def render_problem(state: GraphState, include_category: bool = False) -> str:
    """
    상태에 담긴 문제를 프롬프트용 텍스트로 만든다.

    예시는 입력과 출력만 넣고 설명은 제외한다.

    Parameters:
        state (GraphState): 문제 필드가 채워진 상태
        include_category (bool): 카테고리와 선정 이유 포함 여부

    Returns:
        str: 프롬프트에 넣을 문제 텍스트
    """
    lines = []
    if include_category:
        lines.append(f"카테고리: {state.category} ({state.category_select_reason})")

    lines += [
        f"제목: {state.problem_title}",
        f"지문:\n{state.problem_content}",
        f"입력 형식:\n{state.input_format}",
        f"출력 형식:\n{state.output_format}",
        "제약:",
        *[render_constraint(c) for c in state.input_constraints],
        "예시:",
    ]
    for idx, example in enumerate(state.problem_examples, start=1):
        lines.append(f"[예시 {idx}]\n입력:\n{example.input}\n출력:\n{example.output}")

    return "\n".join(lines)
