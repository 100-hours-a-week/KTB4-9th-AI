"""
문제 생성 노드를 실제 Gemini로 한 번 실행하고 정적 검증 결과를 출력한다.

사용법:
    uv run python -m scripts.try_generate_problem LV2 DP
"""

import asyncio
import sys
import time

from src.problem.nodes.generate_problem import generate_problem
from src.problem.nodes.static_validate import static_validate
from src.problem.state import GraphState

SHOW_FIELDS = {
    "difficulty",
    "category",
    "category_select_reason",
    "problem_title",
    "problem_description",
    "input_format",
    "output_format",
    "problem_examples",
    "input_constraints",
}


async def main(difficulty: str, category: str) -> None:
    state = GraphState(requested_difficulty=difficulty, requested_category=category)

    started = time.perf_counter()
    update = await generate_problem(state)
    elapsed = time.perf_counter() - started

    state = state.model_copy(update=update)
    print(state.model_dump_json(indent=2, include=SHOW_FIELDS))
    print(f"\n생성 시간: {elapsed:.1f}초")

    result = await static_validate(state)
    print(f"정적 검증: {result}")


if __name__ == "__main__":
    requested_difficulty = sys.argv[1] if len(sys.argv) > 1 else "LV2"
    requested_category = sys.argv[2] if len(sys.argv) > 2 else "DP"
    asyncio.run(main(requested_difficulty, requested_category))
