from pydantic import BaseModel, ConfigDict, Field

from src.enum import Category, ConstraintDataType, ConstraintScope, Difficulty, Language


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
    explanation: str


class HiddenTestCase(CamelBaseModel):
    input: str
    output: str


class HintComment(CamelBaseModel):
    language: Language
    comment: str


class SolutionCode(CamelBaseModel):
    language: Language
    code: str


class Problem(CamelBaseModel):
    problem_title: str
    problem_description: str
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

    def get_problem_content(self) -> str:
        return (
            f"{self.problem_description}\n\n"
            f"[입력]\n{self.input_format}\n\n"
            f"[출력]\n{self.output_format}"
        )


class EvaluationRequest(CamelBaseModel):
    problem_title: str | None = None
    problem_description: str | None = None
    category: Category | None = None  # 추후 ENUM으로 교체
    category_select_reason: str | None = None
    solution_keywords: list[str] | None = None
    input_constraints: list[InputConstraint] | None = None
    execution_limits: list[ExecutionLimit] | None = None
    natural_solution: str | None = None


class EvaluationResponse(CamelBaseModel):
    success: bool = True
    message: str | None = None
    score: int | None = None
    llm_feedback: str | None = None
    keywords: list[Keyword] | None = None


class ProblemRequest(CamelBaseModel):
    difficulty: Difficulty
    category: Category


class ProblemResponse(CamelBaseModel):
    success: bool = True
    message: str | None = None
    problem: Problem
