from pydantic import Field, field_validator

from src.core.enums import (
    Category,
    ConstraintDataType,
    ConstraintScope,
    Difficulty,
    Language,
    SeedSource,
)
from src.schema.base import BaseResponse, CamelBaseModel


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


class ProblemRequest(CamelBaseModel):
    difficulty: Difficulty
    category: Category


class ProblemResponse(BaseResponse):
    problem: Problem | None = None


class DailyProblemResponse(BaseResponse):
    problem: Problem | None = None


class BattleProblemResponse(BaseResponse):
    battle_problem: BattleProblem
