from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field

from src.problem.schema import Difficulty


class Language(StrEnum):
    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    CPP = "cpp"


class ConstraintScope(StrEnum):
    INPUT = "input"
    OUTPUT = "output"


class ConstraintDataType(StrEnum):
    INT = "int"
    LONG = "long"
    DOUBLE = "double"
    STRING = "str"
    CHAR = "char"
    BOOLEAN = "bool"


class DiscardReason(StrEnum):
    EMPTY_FIELD = "empty_field"
    EXAMPLE_OUT_OF_RANGE = "example_out_of_range"
    MISMATCH_TYPE = "mismatch_type"
    CONSTRAINT_CONFLICT = "constraint_conflict"
    MISMATCH_LABEL = "mismatch_label"
    DUPLICATE = "duplicate"
    MAX_ATTEMPT_EXCEEDED = "max_attempt_exceeded"
    PROBLEM_CONTRADICTION = "problem_contradiction"
    CONSTRAINT_CONTRADICTION = "constraint_contradiction"


class Example(BaseModel):
    """problem_examples. 문제당 최대 3개 (display_order 1~3)"""

    input: str
    output: str
    explanation: str | None = None


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


class TestCase(BaseModel):
    """test_cases. 문제당 최대 100개 (display_order 1~100)"""

    input: str
    expected_output: str


class HintComment(BaseModel):
    language: Language
    content: str | None = None


class HintSolutionCode(BaseModel):
    language: Language
    content: str | None = None


class LLMConfig(BaseModel):
    model_name: str | None = None
    model_version: str | None = None
    prompt_version: str | None = None
    temperature: float | None = None
    top_k: int | None = 3


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
    problem_description: str | None = None
    input_format: str | None = None
    output_format: str | None = None
    problem_examples: list[Example] = Field(default_factory=list)
    input_constraints: list[InputConstraint] = Field(default_factory=list)
    execution_limits: list[ExecutionLimit] = Field(default_factory=list)

    # ---- testcase
    hidden_test_cases: list[TestCase] = Field(default_factory=list)

    # ---- hint
    hint_comments: list[HintComment] = Field(default_factory=list)
    hint_solution_codes: list[HintSolutionCode] = Field(default_factory=list)

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

    # ---- 폐기 이유
    is_discarded: bool = False
    discard_reason: DiscardReason | None = None
    discard_detail: str | None = None

    # ---- 분석용
    node_models: Annotated[dict[str, LLMConfig], merge_node_models] = Field(
        default_factory=dict
    )
