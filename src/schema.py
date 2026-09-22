from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.enum import (
    Category,
    ConstraintDataType,
    ConstraintScope,
    Difficulty,
    ErrorCode,
    Language,
    SeedSource,
)


def to_camel(string: str) -> str:
    parts = string.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


class CamelBaseModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


class InputConstraint(CamelBaseModel):
    target: str
    scope: ConstraintScope
    data_type: ConstraintDataType
    min_value: float | None = None
    max_value: float | None = None
    data_count: int | None = None
    special_conditions: list[str] = Field(default_factory=list)


class ExecutionLimit(CamelBaseModel):
    language: Language
    time_limit_ms: int
    memory_limit_kb: int


class Keyword(CamelBaseModel):
    keyword: str
    is_included: bool


class ProblemExample(CamelBaseModel):
    input: str
    output: str
    description: str | None = None


class HiddenTestCase(CamelBaseModel):
    input: str
    output: str


class HintComment(CamelBaseModel):
    language: Language
    content: str


class SolutionCode(CamelBaseModel):
    language: Language
    content: str


class Problem(CamelBaseModel):
    problem_title: str
    problem_content: str
    input_format: str
    output_format: str
    requested_difficulty: Difficulty
    difficulty: Difficulty
    category: Category
    category_select_reason: str
    input_constraints: list[InputConstraint]
    execution_limits: list[ExecutionLimit]
    problem_examples: list[ProblemExample]
    solution_keywords: list[str] = Field(default_factory=list)
    hidden_test_cases: list[HiddenTestCase] = Field(default_factory=list)
    hint_comments: list[HintComment] = Field(default_factory=list)
    solution_codes: list[SolutionCode] = Field(default_factory=list)


class BattleProblem(CamelBaseModel):
    category: Category | None = None
    problem_title: str
    problem_content: str
    test_cases: list[HiddenTestCase]


class FewshotSeedCreate(CamelBaseModel):
    source: SeedSource = SeedSource.MANUAL
    source_ref: str | None = None
    category: Category
    difficulty: Difficulty
    problem_title: str
    problem_content: str
    input_format: str
    output_format: str
    input_constraints: list[InputConstraint]
    execution_limits: list[ExecutionLimit]
    problem_examples: list[ProblemExample]

    @field_validator("category")
    @classmethod
    def _no_random(cls, v: Category) -> Category:
        if v == Category.RANDOM:
            raise ValueError("시드 카테고리에 RANDOM은 쓸 수 없습니다")
        return v


class BaseResponse(CamelBaseModel):
    success: bool = True
    message: str | None = None
    error_code: ErrorCode | None = None


class EvaluationRequest(CamelBaseModel):
    problem_title: str | None = None
    problem_content: str | None = None
    category: Category | None = None  # 추후 ENUM으로 교체
    category_select_reason: str | None = None
    solution_keywords: list[str] | None = None
    input_constraints: list[InputConstraint] | None = None
    execution_limits: list[ExecutionLimit] | None = None
    natural_solution: str | None = None


class EvaluationResponse(BaseResponse):
    score: int | None = None
    llm_feedback: str | None = None
    keywords: list[Keyword] | None = None


class ProblemRequest(CamelBaseModel):
    difficulty: Difficulty
    category: Category


class ProblemResponse(BaseResponse):
    problem: Problem | None = None


class DailyProblemResponse(BaseResponse):
    problem: Problem | None = None


class BattleProblemResponse(BaseResponse):
    battle_problem: BattleProblem
