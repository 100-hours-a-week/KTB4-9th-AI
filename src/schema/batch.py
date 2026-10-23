from typing import Any

from pydantic import Field, field_validator

from src.core.enums import Category, Difficulty
from src.schema.base import BaseResponse, CamelBaseModel


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


class PlanItem(CamelBaseModel):
    category: Category
    difficulty: Difficulty
    count: int


class BatchPlan(CamelBaseModel):
    """생성 배치가 지금 돈다면 만들 문제. 아무것도 쓰지 않고 계산만 한다."""

    spring_item_count: int | None  # None이면 Spring 재고를 받지 못함
    pending_normal: int
    pending_daily: int
    pending_battle: int
    items: list[PlanItem]
    total: int


class BatchPlanResponse(BaseResponse, BatchPlan):
    pass
