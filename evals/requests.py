"""평가용 고정 요청 목록.

모델을 비교할 때 모든 모델이 같은 요청을 같은 순서로 받아야 한다.
목록은 한 번 만들어 파일로 커밋하고 고치지 않는다. 바꿀 일이 생기면 v2를 만든다.

requests_v1.jsonl: LV1~3 × 12카테고리 × 2건 = 72건, 순서는 섞여 있다.
- 제외: MATH(프론트에서 뺌), ARRAY·STACK_QUEUE·HEAP(학습 데이터에 없음), RANDOM
"""

from pathlib import Path

from pydantic import BaseModel

from src.core.enums import Category, Difficulty


class EvalRequest(BaseModel):
    """요청 한 건. id로 모델별 결과를 짝짓는다."""

    id: str
    difficulty: Difficulty
    category: Category
    seed: int  # 실행 직전 random.seed()에 넣는다


def load_requests(path: Path) -> list[EvalRequest]:
    """
    JSONL 파일에서 요청 목록을 읽는다.

    Parameters:
        path (Path): 요청 목록 파일

    Returns:
        list[EvalRequest]: 파일에 적힌 순서 그대로의 요청
    """
    with path.open(encoding="utf-8") as file:
        return [EvalRequest.model_validate_json(line) for line in file if line.strip()]
