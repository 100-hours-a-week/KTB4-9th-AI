"""의미 검증. 레퍼런스 코드를 공개 예시에 돌려 출력을 맞춰 본다."""

import logging

from src.client.judge0 import run_code
from src.core.enums import DiscardReason, ExecutionStatus
from src.problem.nodes.generate_ref_code import REFERENCE_LANGUAGE
from src.problem.state import (
    ExecutionLimit,
    GraphState,
    discard,
    find_limit,
)

logger = logging.getLogger(__name__)


def normalize_output(text: str) -> str:
    """
    출력을 비교할 수 있는 모양으로 만든다.

    줄 끝 공백과 끝의 빈 줄을 무시한다. 채점기가 보통 이렇게 비교한다.

    Parameters:
        text (str): 표준 출력 또는 기대 출력

    Returns:
        str: 비교용으로 정리한 문자열
    """
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


async def check_examples(
    state: GraphState, limit: ExecutionLimit
) -> tuple[DiscardReason, str] | None:
    """
    공개 예시마다 레퍼런스 코드를 돌려 출력을 맞춰 본다.

    Parameters:
        state (GraphState): reference_code와 공개 예시가 채워진 상태
        limit (ExecutionLimit): 레퍼런스 언어의 실행 제한

    Returns:
        tuple[DiscardReason, str] | None: 어긋난 첫 예시의 사유. 모두 맞으면 None
    """
    for index, example in enumerate(state.problem_examples, start=1):
        result = await run_code(
            state.reference_code or "", REFERENCE_LANGUAGE, example.input, limit
        )
        if not result.succeeded:
            tail = result.stderr.strip().splitlines()
            detail = f" ({tail[-1][:80]})" if tail else ""
            reason = (
                DiscardReason.EXECUTOR_ERROR
                if result.status is ExecutionStatus.INTERNAL_ERROR
                else DiscardReason.REFERENCE_CODE_FAILED
            )
            return reason, f"예시 {index}: {result.status.value}{detail}"

        actual = normalize_output(result.stdout)
        expected = normalize_output(example.output)
        if actual != expected:
            return (
                DiscardReason.EXAMPLE_MISMATCH,
                f"예시 {index}: 기대 {expected!r} / 실행 결과 {actual!r}",
            )
    return None


async def semantic_validate(state: GraphState) -> dict:
    """
    레퍼런스 코드가 공개 예시를 재현하는지 검사한다.

    실행 기반 검사가 실패하면 레퍼런스 코드를 다시 만들어 재시도한다
    (route_after_semantic이 generate_ref_code로 되돌린다).
    한도를 다 쓰면 실패한 실제 사유로 폐기한다.

    Parameters:
        state (GraphState): 검증용 코드까지 만들어진 상태

    Returns:
        dict: 검증 결과. 실패 시 재시도 표시 또는 폐기 정보
    """
    attempt = state.semantic_validation_attempt + 1
    updates: dict = {"semantic_validation_attempt": attempt}

    if not state.reference_code:
        return updates | discard(DiscardReason.EMPTY_FIELD, "검증용 코드가 비었음")
    if not state.problem_examples:
        return updates | discard(DiscardReason.EMPTY_FIELD, "공개 예시가 없음")

    limit = find_limit(state.execution_limits, REFERENCE_LANGUAGE)
    if limit is None:
        return updates | discard(
            DiscardReason.EMPTY_FIELD, f"{REFERENCE_LANGUAGE.value} 실행 제한 없음"
        )

    failure = await check_examples(state, limit)
    if failure is None:
        return updates | {"is_semantically_valid": True}

    reason, detail = failure
    if reason is DiscardReason.EXECUTOR_ERROR:
        # 채점 서버 장애는 레퍼런스 코드를 다시 만들어도 풀리지 않는다.
        # 재시도하면 LLM 호출만 낭비하므로 바로 폐기한다.
        logger.warning("의미 검증 실행기 오류로 폐기: %s", detail)
        return updates | discard(reason, detail)

    if attempt < state.semantic_validation_max_attempt:
        # 레퍼런스 코드를 다시 만들어 본다
        logger.info("의미 검증 재시도 %d회: %s", attempt, detail)
        return updates | {"is_semantically_valid": False}

    return updates | discard(reason, f"{detail} ({attempt}회 시도)")
