import uuid
from typing import Annotated

from pydantic import BaseModel, Field

from src.client.llm import LLMConfig
from src.core.enums import (
    ConstraintDataType,
    ConstraintScope,
    Difficulty,
    DiscardReason,
    Language,
    ProblemPurpose,
    Trigger,
)

HIDDEN_TEST_CASE_COUNT = 7  # 문제당 보관할 비공개 테스트 케이스 수


class ProblemExample(BaseModel):
    """problem_examples. 문제당 최대 3개 (display_order 1~3)"""

    input: str
    output: str
    description: str | None = None


class InputConstraint(BaseModel):
    """problems.constraints (JSON, 화면 표시용)"""

    target: str
    scope: ConstraintScope
    data_type: ConstraintDataType
    # data_type이 숫자형일 때만 값이 있다. 문자열·문자·불이면 비워 둔다.
    min_value: float | None = None
    max_value: float | None = None
    data_count: int | None = None
    special_conditions: list[str] = Field(default_factory=list)


class ExecutionLimit(BaseModel):
    """running_limits. 문제·언어별 한 행"""

    language: Language
    time_limit_ms: int
    memory_limit_kb: int


class HiddenTestCase(BaseModel):
    """test_cases. 문제당 HIDDEN_TEST_CASE_COUNT개 (display_order 1부터)"""

    input: str
    output: str


class HintComment(BaseModel):
    language: Language
    content: str | None = None


class SolutionCode(BaseModel):
    language: Language
    content: str | None = None


def find_limit(
    limits: list[ExecutionLimit], language: Language
) -> ExecutionLimit | None:
    """언어에 해당하는 실행 제한을 찾는다. 없으면 None."""
    return next((limit for limit in limits if limit.language is language), None)


def keep_discard_flag(left: bool, right: bool) -> bool:
    """리듀서가 없으면 병렬 노드 둘이 동시에 폐기할 때 그래프가 죽는다.

    폐기는 되돌리지 않으므로 한 번 True가 되면 그대로 둔다.
    """
    return left or right


def keep_first_discard[T](left: T | None, right: T | None) -> T | None:
    """먼저 기록된 폐기 정보를 남긴다.

    reason과 detail에 같은 규칙을 써야 둘이 짝을 유지한다.
    같은 단계에서 다른 노드도 실패했다면 그 사유는 버려진다.
    """
    return left if left is not None else right


def merge_node_models(
    left: dict[str, LLMConfig],
    right: dict[str, LLMConfig],
) -> dict[str, LLMConfig]:
    """리듀서가 없으면 병렬 노드가 node_models를 동시에 덮어써 충돌한다."""
    return {**left, **right}


def discard(reason: DiscardReason, detail: str) -> dict:
    """폐기 처리 공통 반환값."""
    return {
        "is_discarded": True,
        "discard_reason": reason,
        "discard_detail": detail,
    }


class GraphState(BaseModel):
    # ---- input
    requested_difficulty: Difficulty
    requested_category: str
    # 노드는 읽지 않는다. finalize가 저장할 때 전송 대상을 가르는 데만 쓴다.
    trigger: Trigger = Trigger.ON_DEMAND
    purpose: ProblemPurpose = ProblemPurpose.NORMAL

    # ---- problem
    difficulty: Difficulty | None = None
    category: str | None = None
    category_select_reason: str | None = None
    problem_title: str | None = None
    problem_content: str | None = None
    input_format: str | None = None
    output_format: str | None = None
    problem_examples: list[ProblemExample] = Field(default_factory=list)
    input_constraints: list[InputConstraint] = Field(default_factory=list)
    execution_limits: list[ExecutionLimit] = Field(default_factory=list)

    # ---- testcase
    hidden_test_cases: list[HiddenTestCase] = Field(default_factory=list)

    # ---- 정답 코드와 힌트 (generate_solution_code가 함께 채운다)
    solution_codes: list[SolutionCode] = Field(default_factory=list)
    hint_comments: list[HintComment] = Field(default_factory=list)

    # ---- nl keywords
    solution_keywords: list[str] = Field(default_factory=list)

    # ---- 정적 검증
    is_statically_validated: bool = False

    # ---- 중복 검사
    algorithm_core: str | None = None
    # check_duplicate가 만들어 finalize가 색인에 다시 쓴다. 두 번 부르지 않는다.
    algorithm_core_embedding: list[float] | None = None

    # ---- 검증용 예시 코드
    reference_code: str | None = None
    reference_language: Language | None = None

    # ---- 의미 검증
    is_semantically_valid: bool = False
    semantic_validation_attempt: int = 0
    semantic_validation_max_attempt: int = 3

    # ---- 폐기 이유 (병렬 노드가 동시에 써도 되도록 리듀서를 둔다)
    is_discarded: Annotated[bool, keep_discard_flag] = False
    discard_reason: Annotated[DiscardReason | None, keep_first_discard] = None
    discard_detail: Annotated[str | None, keep_first_discard] = None
    # 어느 노드에서 걸렸는지. build_graph가 노드마다 자동으로 채운다.
    discard_stage: Annotated[str | None, keep_first_discard] = None

    # ---- 저장 결과
    problem_id: uuid.UUID | None = None  # finalize가 채운다

    # ---- 분석용
    node_models: Annotated[dict[str, LLMConfig], merge_node_models] = Field(
        default_factory=dict
    )
