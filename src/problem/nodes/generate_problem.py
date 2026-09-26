import logging
import random

from pydantic import BaseModel, Field

from src.client.llm import LLMConfig, call_llm_structured
from src.core.enums import Category, Difficulty, Language
from src.db.repository import FewshotSeedRepository
from src.db.session import session_scope
from src.problem.state import (
    ExecutionLimit,
    GraphState,
    InputConstraint,
    ProblemExample,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-3.5-flash-lite"
PROMPT_VERSION = "generate_problem/v4"

GENERATE_PROBLEM_PROMPT = """당신은 코딩 테스트 문제 출제자다.

[요청]
난이도: {difficulty}
카테고리: {category} (category에 그대로 적는다)

[참고 문제] (형식과 난이도 감각만 참고하고 내용을 복제하지 않는다)
{few_shot}

[규칙]
- 요청 난이도와 카테고리에 맞는 새로운 문제를 한국어로 만든다.
- difficulty에는 요청 난이도 {difficulty}를 그대로 적는다.
- 공개 예시는 1~3개이며, 모든 예시는 제약 조건을 만족해야 한다.
- 예시의 output은 input을 실제로 풀었을 때 나오는 정확한 값이어야 한다.
- input_constraints에는 입력 항목(scope: input)과
  출력(scope: output, target: output)을 모두 적는다.
- data_type이 숫자형(int, long, double)일 때만 min_value, max_value를 숫자로 적는다.
- 숫자형이 아니면(str, char, bool) min_value, max_value를 비워 둔다.
- 값들이 서로 달라야 하면 special_conditions에 "서로 다른 값"이라고 적는다.
- execution_limits는 {languages} 네 언어를 모두 적는다.
- category_select_reason에는 이 카테고리로 판단한 근거를 한 문장으로 적는다.
- algorithm_core에는 입출력 형식과 이야기 설정을 빼고,
  어떤 자료구조·알고리즘으로 무엇을 계산하는지 한 문장으로 적는다.
"""

RESPONSE_CATEGORIES = [c for c in Category if c != Category.RANDOM]


class GeneratedProblem(BaseModel):
    """문제 생성 응답 형식. 모델은 이 구조와 enum 값으로만 응답한다."""

    difficulty: Difficulty
    category: Category
    category_select_reason: str
    algorithm_core: str = Field(description="입출력 형식을 뺀 핵심 풀이 한 문장")
    problem_title: str
    problem_content: str
    input_format: str
    output_format: str
    problem_examples: list[ProblemExample]
    input_constraints: list[InputConstraint]
    execution_limits: list[ExecutionLimit]


def resolve_category(requested: str) -> Category:
    """
    생성할 카테고리를 정한다.

    RANDOM이면 응답 카테고리 중 하나를 고른다.

    Parameters:
        requested (str): 요청 카테고리

    Returns:
        Category: RANDOM이 아닌 실제 카테고리
    """
    if requested == Category.RANDOM:
        return random.choice(RESPONSE_CATEGORIES)
    return Category(requested)


async def fetch_few_shot(
    difficulty: Difficulty, category: Category, k: int
) -> list[dict]:
    """
    같은 난이도·카테고리의 활성 few-shot 시드를 무작위로 조회한다.

    참고 문제는 선택 사항이므로 조회에 실패해도 빈 목록으로 진행한다.

    Parameters:
        difficulty (Difficulty): 요청 난이도
        category (Category): 생성할 카테고리
        k (int): 조회할 문제 수

    Returns:
        list[dict]: 참고 문제 목록
    """
    if category == Category.RANDOM:
        return []
    try:
        async with session_scope() as session:
            seeds = await FewshotSeedRepository(session).sample(category, difficulty, k)
    except Exception as error:
        logger.warning(
            "few-shot 조회 실패, 참고 문제 없이 진행: %s", error, exc_info=True
        )
        return []
    return [
        {
            "problem_title": seed.problem_title,
            "problem_content": seed.problem_content,
            "input_format": seed.input_format,
            "output_format": seed.output_format,
        }
        for seed in seeds
    ]


def render_few_shot(problems: list[dict]) -> str:
    """참고 문제 목록을 프롬프트용 텍스트로 만든다."""
    if not problems:
        return "(참고 문제 없음)"
    blocks = []
    for idx, problem in enumerate(problems, start=1):
        blocks.append(
            f"[예시 {idx}]\n"
            f"제목: {problem.get('problem_title')}\n"
            f"지문: {problem.get('problem_content')}\n"
            f"입력 형식: {problem.get('input_format')}\n"
            f"출력 형식: {problem.get('output_format')}"
        )
    return "\n\n".join(blocks)


def build_prompt(
    difficulty: Difficulty, category: Category, few_shot: list[dict]
) -> str:
    """
    문제 생성 프롬프트를 만든다.

    Parameters:
        difficulty (Difficulty): 요청 난이도
        category (Category): 생성할 카테고리
        few_shot (list[dict]): 참고 문제 목록

    Returns:
        str: 완성된 프롬프트
    """
    return GENERATE_PROBLEM_PROMPT.format(
        difficulty=difficulty.value,
        category=category.value,
        few_shot=render_few_shot(few_shot),
        languages=", ".join(lang.value for lang in Language),
    )


async def generate_problem(state: GraphState) -> dict:
    """
    요청 난이도·카테고리로 문제 지문, 제약, 공개 예시, 핵심 풀이 아이디어를 생성한다.

    카테고리는 resolve_category가 정한 값을 쓴다. 요청이 RANDOM이어도
    requested_category는 그대로 두어 폐기 로그에 요청 값이 남게 한다.

    Parameters:
        state (GraphState): 요청 난이도·카테고리가 담긴 상태

    Returns:
        dict: 생성한 문제 필드와 호출 설정

    Raises:
        LLMOutputParseError: 응답이 형식에 맞지 않는 경우
    """
    # temperature는 넘기지 않는다. 이 모델은 고정 샘플링이라 값을 무시한다.
    cfg = LLMConfig(
        model_name=MODEL_NAME,
        prompt_version=PROMPT_VERSION,
        top_k=3,
    )

    category = resolve_category(state.requested_category)
    few_shot = await fetch_few_shot(state.requested_difficulty, category, cfg.top_k)
    prompt = build_prompt(state.requested_difficulty, category, few_shot)
    problem = await call_llm_structured(prompt, cfg, GeneratedProblem)

    if problem.category != category:
        # 드문 경우라 생성 결과를 버리지 않고 코드가 정한 카테고리를 쓴다.
        logger.warning(
            "응답 카테고리 불일치: 요청 %s, 응답 %s", category, problem.category
        )

    return {
        "difficulty": problem.difficulty,
        "category": category.value,
        "category_select_reason": problem.category_select_reason,
        "algorithm_core": problem.algorithm_core,
        "problem_title": problem.problem_title,
        "problem_content": problem.problem_content,
        "input_format": problem.input_format,
        "output_format": problem.output_format,
        "problem_examples": problem.problem_examples,
        "input_constraints": problem.input_constraints,
        "execution_limits": problem.execution_limits,
        "node_models": {"generate_problem": cfg},
    }
