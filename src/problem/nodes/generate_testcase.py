import asyncio
import logging

from pydantic import BaseModel, Field

from src.enum import DiscardReason
from src.problem.nodes.generate_ref_code import REFERENCE_LANGUAGE
from src.problem.render import render_problem
from src.problem.state import (
    HIDDEN_TEST_CASE_COUNT,
    ExecutionLimit,
    GraphState,
    HiddenTestCase,
    LLMConfig,
    ProblemExample,
    discard,
    find_limit,
)
from src.shared.code_runner import RunResult, run_code
from src.shared.llm import call_llm_structured

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-3.7-flash"
PROMPT_VERSION = "generate_testcase/v2"

# 중복과 공개 예시 겹침으로 줄어드는 몫을 감안해 넉넉히 요청한다.
# 정확히 필요한 수만 요청하면 하나만 겹쳐도 문제 전체가 폐기된다.
REQUEST_MARGIN = 3
REQUESTED_COUNT = HIDDEN_TEST_CASE_COUNT + REQUEST_MARGIN

GENERATE_TESTCASE_PROMPT = """당신은 코딩 테스트 문제의 검증자다.

[문제]
{problem}

[규칙]
- 위 문제의 비공개 테스트 케이스 입력 {count}개를 만든다.
- 모든 입력은 입력 형식과 제약 조건을 정확히 지킨다.
- 다음이 골고루 섞이게 한다.
  - 제약의 최솟값과 최댓값에 걸친 경계 입력
  - 흔한 크기의 보통 입력
  - 틀린 풀이가 걸리기 쉬운 입력
    (중복 값, 음수, 이미 정렬된 입력, 원소가 하나뿐인 입력 등)
- 같은 입력을 두 번 넣지 않는다.
- 공개 예시에 나온 입력은 넣지 않는다.
- input에는 그 테스트 케이스의 표준 입력 전문만 넣는다. 설명을 넣지 않는다.
"""


class HiddenTestCaseInputs(BaseModel):
    """비공개 테스트 케이스 입력 응답 형식.

    기대 출력은 모델에게 묻지 않는다. 사람이 암산으로 20문제를 푸는 것과
    같아서 틀린 값이 섞이고, 그 값으로 채점하면 맞는 답이 틀렸다고 나온다.
    출력은 레퍼런스 코드를 실제로 실행해서 얻는다.
    """

    inputs: list[str] = Field(description="표준 입력 전문 목록")


def build_prompt(state: GraphState) -> str:
    """
    비공개 테스트 케이스 입력 생성 프롬프트를 만든다.

    카테고리는 넣지 않는다. 입력을 만드는 데는 입력 형식과 제약만 필요하다.

    Parameters:
        state (GraphState): 문제 필드가 채워진 상태

    Returns:
        str: 완성된 프롬프트
    """
    return GENERATE_TESTCASE_PROMPT.format(
        problem=render_problem(state, include_category=False),
        count=REQUESTED_COUNT,
    )


def normalize_inputs(inputs: list[str], examples: list[ProblemExample]) -> list[str]:
    """
    앞뒤 공백과 빈 값을 걷어내고, 중복과 공개 예시 입력을 없앤다.

    공개 예시와 같은 입력은 비공개 테스트 케이스로서 값이 없다.

    Parameters:
        inputs (list[str]): 모델이 준 입력 목록
        examples (list[ProblemExample]): 공개 예시

    Returns:
        list[str]: 정리한 입력 목록. HIDDEN_TEST_CASE_COUNT개까지만
    """
    seen = {example.input.strip() for example in examples}
    cleaned: list[str] = []
    for value in inputs:
        text = value.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned[:HIDDEN_TEST_CASE_COUNT]


# 폐기 사유에 실패를 몇 건까지 적을지
MAX_REPORTED_FAILURES = 3


def summarize_failures(failures: list[str]) -> str:
    """실패를 몇 건만 적고 나머지는 개수로 줄인다."""
    shown = "; ".join(failures[:MAX_REPORTED_FAILURES])
    remaining = len(failures) - MAX_REPORTED_FAILURES
    return f"{shown} 외 {remaining}건" if remaining > 0 else shown


def describe_failure(index: int, result: RunResult | BaseException) -> str | None:
    """한 입력의 실행 결과에서 실패 사유를 뽑는다. 성공이면 None."""
    if isinstance(result, BaseException):
        return f"입력 {index}: 실행기 오류 {result}"
    if not result.succeeded:
        detail = result.stderr.strip().splitlines()
        tail = f" ({detail[-1][:80]})" if detail else ""
        return f"입력 {index}: {result.status.value}{tail}"
    if not result.stdout.rstrip():
        # 정답 코드가 아무것도 출력하지 않으면 채점에 쓸 수 없다
        return f"입력 {index}: 출력이 비었음"
    return None


async def run_reference(
    state: GraphState, inputs: list[str], limit: ExecutionLimit
) -> list[RunResult | BaseException]:
    """레퍼런스 코드를 입력마다 돌린다. 동시 실행 수는 실행기가 묶는다."""
    return await asyncio.gather(
        *(
            run_code(state.reference_code or "", REFERENCE_LANGUAGE, value, limit)
            for value in inputs
        ),
        return_exceptions=True,
    )


async def generate_testcase(state: GraphState) -> dict:
    """
    비공개 테스트 케이스를 만든다.

    입력은 모델이 만들고, 기대 출력은 레퍼런스 코드를 실행해서 채운다.

    병렬 노드이므로 예외를 올리지 않는다. 올리면 그래프 전체가 죽는다.

    Parameters:
        state (GraphState): 의미 검증을 통과한 상태

    Returns:
        dict: 비공개 테스트 케이스와 호출 설정. 실패 시 폐기 정보
    """
    cfg = LLMConfig(
        model_name=MODEL_NAME,
        prompt_version=PROMPT_VERSION,
        temperature=0.7,  # 입력이 서로 다양해야 한다
    )

    if not state.reference_code:
        return discard(DiscardReason.EMPTY_FIELD, "검증용 코드가 비었음")

    limit = find_limit(state.execution_limits, REFERENCE_LANGUAGE)
    if limit is None:
        return discard(
            DiscardReason.EMPTY_FIELD,
            f"{REFERENCE_LANGUAGE.value} 실행 제한 없음",
        )

    try:
        response = await call_llm_structured(
            build_prompt(state), cfg, HiddenTestCaseInputs
        )
    except Exception as error:
        # 응답 형식 오류뿐 아니라 키 누락, 타임아웃, 레이트 리밋도 여기로 온다
        logger.warning("테스트 케이스 생성 실패: %s", error, exc_info=True)
        return discard(DiscardReason.LLM_ERROR, f"테스트 케이스 생성 실패: {error}")

    inputs = normalize_inputs(response.inputs, state.problem_examples)
    if len(inputs) < HIDDEN_TEST_CASE_COUNT:
        return discard(
            DiscardReason.LLM_ERROR,
            f"테스트 케이스가 {HIDDEN_TEST_CASE_COUNT}개 미만: {len(inputs)}개",
        )

    results = await run_reference(state, inputs, limit)
    failures = [
        message
        for index, result in enumerate(results, start=1)
        if (message := describe_failure(index, result)) is not None
    ]
    if failures:
        # 한 입력이라도 실패하면 레퍼런스 코드나 제약이 잘못된 것이다.
        # 그 상태로 만든 테스트 케이스는 채점에 쓸 수 없으므로 문제를 버린다.
        summary = summarize_failures(failures)
        logger.warning("레퍼런스 코드 실행 실패: %s", summary)
        return discard(
            DiscardReason.REFERENCE_CODE_FAILED, f"레퍼런스 코드 실행 실패: {summary}"
        )

    return {
        "hidden_test_cases": [
            HiddenTestCase(input=value, output=result.stdout.rstrip())
            for value, result in zip(inputs, results, strict=True)
        ],
        "node_models": {"generate_testcase": cfg},
    }
