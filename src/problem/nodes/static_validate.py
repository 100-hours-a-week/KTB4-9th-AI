import re

from src.problem.state import (
    ConstraintDataType,
    ConstraintScope,
    DiscardReason,
    GraphState,
    Language,
    discard,
)

NUMERIC_TYPES = {
    ConstraintDataType.INT,
    ConstraintDataType.LONG,
    ConstraintDataType.DOUBLE,
}
INTEGER_TYPES = {ConstraintDataType.INT, ConstraintDataType.LONG}


def to_number(text: str | None) -> float | None:
    """
    제약 값 문자열을 숫자로 변환한다.

    "100000", "-200000", "1e5", "10^5", "2*10^5", "100,000" 형식을 허용한다.

    Parameters:
        text (str | None): 제약 값 문자열

    Returns:
        float | None: 변환한 숫자. 변환할 수 없으면 None
    """
    if text is None:
        return None
    cleaned = text.replace(",", "").replace(" ", "")
    sign = -1 if cleaned.startswith("-") else 1
    body = cleaned.lstrip("-")

    power = re.fullmatch(r"(?:(\d+(?:\.\d+)?)\*)?(\d+)\^(\d+)", body)
    if power:
        coef = float(power.group(1)) if power.group(1) else 1.0
        return sign * coef * float(power.group(2)) ** int(power.group(3))

    try:
        return float(cleaned)
    except ValueError:
        return None


def matches_type(token: str, data_type: ConstraintDataType) -> bool:
    """
    출력 토큰 하나가 지정한 자료형으로 해석되는지 확인한다.

    Parameters:
        token (str): 공백으로 나눈 출력 값 하나
        data_type (ConstraintDataType): 기대 자료형

    Returns:
        bool: 자료형이 맞으면 True
    """
    if data_type in INTEGER_TYPES:
        return re.fullmatch(r"-?\d+", token) is not None
    if data_type == ConstraintDataType.DOUBLE:
        return to_number(token) is not None
    if data_type == ConstraintDataType.CHAR:
        return len(token) == 1
    if data_type == ConstraintDataType.BOOLEAN:
        return token.lower() in {"true", "false", "0", "1"}
    return bool(token)


async def static_validate(state: GraphState) -> dict:
    """
    LLM 없이 규칙만으로 생성된 문제를 검사한다.

    Parameters:
        state (GraphState): generate_problem 결과가 채워진 상태

    Returns:
        dict: 통과 시 is_statically_validated=True, 실패 시 폐기 정보
    """
    # 1. 필수 필드 누락
    required = {
        "난이도": state.difficulty,
        "카테고리": state.category,
        "문제 제목": state.problem_title,
        "문제 지문": state.problem_description,
        "입력 포맷": state.input_format,
        "출력 포맷": state.output_format,
        "예시": state.problem_examples,
        "제약": state.input_constraints,
        "실행 제한": state.execution_limits,
    }
    for name, value in required.items():
        if not value:
            return discard(DiscardReason.EMPTY_FIELD, f"{name}이(가) 비었음")

    # 2. 언어별 실행 제한 누락
    given = {limit.language for limit in state.execution_limits}
    missing = set(Language) - given
    if missing:
        names = ", ".join(sorted(lang.value for lang in missing))
        return discard(DiscardReason.EMPTY_FIELD, f"실행 제한 없음: {names}")

    # 3. 요청 난이도와 생성 난이도 불일치
    if state.difficulty != state.requested_difficulty:
        return discard(
            DiscardReason.MISMATCH_LABEL,
            f"요청 {state.requested_difficulty.value} / 생성 {state.difficulty.value}",
        )

    # 4. 예시 출력 자료형 불일치
    output_constraints = [
        c for c in state.input_constraints if c.scope == ConstraintScope.OUTPUT
    ]
    for idx, example in enumerate(state.problem_examples, start=1):
        tokens = example.output.split()
        if not tokens:
            return discard(DiscardReason.MISMATCH_TYPE, f"예시 {idx}: 출력이 비었음")
        for constraint in output_constraints:
            bad = [t for t in tokens if not matches_type(t, constraint.data_type)]
            if bad:
                return discard(
                    DiscardReason.MISMATCH_TYPE,
                    f"예시 {idx}: 출력 '{bad[0]}'이(가) "
                    f"{constraint.data_type.value}이(가) 아님",
                )

    # 5. 제약 충돌
    for constraint in state.input_constraints:
        if constraint.data_type not in NUMERIC_TYPES:
            continue
        low = to_number(constraint.min_value)
        high = to_number(constraint.max_value)
        if low is None or high is None:
            continue

        if low > high:
            return discard(
                DiscardReason.CONSTRAINT_CONFLICT,
                f"{constraint.target}: 최솟값 {constraint.min_value}이(가) "
                f"최댓값 {constraint.max_value}보다 큼",
            )

        is_distinct = "서로 다른" in " ".join(constraint.special_conditions)
        if (
            is_distinct
            and constraint.data_type in INTEGER_TYPES
            and constraint.data_count is not None
        ):
            capacity = int(high) - int(low) + 1
            if constraint.data_count > capacity:
                return discard(
                    DiscardReason.CONSTRAINT_CONFLICT,
                    f"{constraint.target}: 서로 다른 값 {constraint.data_count}개를 "
                    f"{capacity}개 범위에서 만들 수 없음",
                )

    return {"is_statically_validated": True}
