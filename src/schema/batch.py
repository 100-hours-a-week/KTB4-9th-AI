from typing import Any

from pydantic import Field, field_validator

from src.core.enums import Category, Difficulty
from src.schema.base import CamelBaseModel


class ProblemDemand(CamelBaseModel):
    """Spring이 알려 주는 조합별 재고. 아직 아무도 풀지 않은 문제 수."""

    difficulty: Difficulty
    category: Category
    count: int = Field(ge=0)

    @field_validator("category")
    @classmethod
    def _no_random(cls, v: Category) -> Category:
        if v == Category.RANDOM:
            raise ValueError("재고 카테고리에 RANDOM은 올 수 없습니다")
        return v


class ProblemDemandData(CamelBaseModel):
    # 항목 하나가 잘못됐다고 전체를 버리지 않도록 원본으로 받아 하나씩 검증한다.
    problem_counts: list[dict[str, Any]]


class ProblemDemandResponse(CamelBaseModel):
    """GET /problems/new 응답."""

    message: str | None = None
    data: ProblemDemandData
