from typing import Annotated

from pydantic import BaseModel, Field

from src.enum import (
    ConstraintDataType,
    ConstraintScope,
    Difficulty,
    DiscardReason,
    Language,
)

HIDDEN_TEST_CASE_COUNT = 20


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
    min_value: str | None = None
    max_value: str | None = None
    data_count: int | None = None
    special_conditions: list[str] = Field(default_factory=list)


class ExecutionLimit(BaseModel):
    """running_limits. 문제·언어별 한 행"""

    language: Language
    time_limit_ms: int
    memory_limit_kb: int


class HiddenTestCase(BaseModel):
    """test_cases. 문제당 HIDDEN_TEST_CASE_COUNT개 (display_order 1~20)"""

    input: str
    output: str


class HintComment(BaseModel):
    language: Language
    content: str | None = None


class SolutionCode(BaseModel):
    language: Language
    content: str | None = None


class LLMConfig(BaseModel):
    model_name: str | None = None
    model_version: str | None = None
    prompt_version: str | None = None
    temperature: float | None = None
    top_k: int | None = 3


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
    is_duplicated: bool = False
    algorithm_core: str | None = None

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

    # ---- 분석용
    node_models: Annotated[dict[str, LLMConfig], merge_node_models] = Field(
        default_factory=dict
    )
