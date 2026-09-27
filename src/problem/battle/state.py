from typing import Annotated

from pydantic import BaseModel, Field

from src.client.llm import LLMConfig
from src.core.enums import Category, DiscardReason, Language
from src.schema.problem import ExecutionLimit

# 배틀 테스트케이스 개수. 명세에 3개로 고정되어 있다.
TEST_CASE_COUNT = 3

MAX_TITLE_LENGTH = 30
MAX_CONTENT_LENGTH = 1000

# 검증용 코드를 실행할 언어. 사용자에게 제공하지 않는다.
REFERENCE_LANGUAGE = Language.PYTHON
REFERENCE_LIMIT = ExecutionLimit(
    language=REFERENCE_LANGUAGE, time_limit_ms=2000, memory_limit_kb=262144
)


def merge_node_models(
    left: dict[str, LLMConfig], right: dict[str, LLMConfig]
) -> dict[str, LLMConfig]:
    """리듀서가 없으면 노드마다 node_models 전체가 덮어써진다."""
    return {**left, **right}


def discard(reason: DiscardReason, stage: str, detail: str) -> dict:
    """폐기 처리 공통 반환값."""
    return {
        "is_discarded": True,
        "discard_reason": reason,
        "discard_stage": stage,
        "discard_detail": detail,
    }


class BattleTestCase(BaseModel):
    """배틀 테스트케이스. 생성 직후에는 output이 비어 있다."""

    input: str
    output: str = ""


class BattleState(BaseModel):
    """배틀 문제 생성 파이프라인의 노드 간 공유 상태."""

    # ---- 문제
    category: Category | None = None
    problem_title: str | None = None
    problem_content: str | None = None
    test_cases: list[BattleTestCase] = Field(default_factory=list)

    # ---- 중복 검사
    algorithm_core: str | None = None
    # check_duplicate가 만들어 finalize가 색인에 다시 쓴다. 두 번 부르지 않는다.
    algorithm_core_embedding: list[float] | None = None

    # ---- 검증용 (응답에 나가지 않는다)
    reference_code: str | None = None

    # ---- 재시도
    fill_attempt: int = 0
    max_fill_attempt: int = 3

    # ---- 폐기
    is_discarded: bool = False
    discard_reason: DiscardReason | None = None
    discard_stage: str | None = None
    discard_detail: str | None = None

    # ---- 분석용
    node_models: Annotated[dict[str, LLMConfig], merge_node_models] = Field(
        default_factory=dict
    )
