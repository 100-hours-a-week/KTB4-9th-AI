from src.core.enums import Category
from src.schema.base import BaseResponse, CamelBaseModel
from src.schema.problem import ExecutionLimit, InputConstraint, Keyword


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
