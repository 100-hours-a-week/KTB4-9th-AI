from src.problem.schema import Category, Difficulty
from src.problem.state import (
    Example,
    ExecutionLimit,
    GraphState,
    InputConstraint,
    Language,
    LLMConfig,
)
from src.shared.llm import LLMOutputParseError, call_llm, parse_json

MODEL_NAME = "gemini-3.5-flash-lite"
PROMPT_VERSION = "generate_problem/v2"

GENERATE_PROBLEM_PROMPT = """당신은 코딩 테스트 문제 출제자다.

[요청]
난이도: {difficulty}
카테고리: {category_instruction}

[참고 문제] (형식과 난이도 감각만 참고하고 내용을 복제하지 않는다)
{few_shot}

[규칙]
- 요청 난이도와 카테고리에 맞는 새로운 문제를 한국어로 만든다.
- difficulty에는 요청 난이도 {difficulty}를 그대로 적는다.
- 공개 예시는 1~3개이며, 모든 예시는 제약 조건을 만족해야 한다.
- 예시의 output은 input을 실제로 풀었을 때 나오는 정확한 값이어야 한다.
- input_constraints에는 입력 항목(scope: input)과 출력(scope: output, target: output)을 모두 적는다.
- data_type은 int, long, double, str, char, bool 중 하나다.
- min_value, max_value는 숫자 문자열로 적는다. 문자열 자료형이면 null이다.
- 값들이 서로 달라야 하면 special_conditions에 "서로 다른 값"이라고 적는다.
- execution_limits는 {languages} 네 언어를 모두 적는다.
- category_select_reason에는 이 카테고리로 판단한 근거를 한 문장으로 적는다.
- algorithm_core에는 입출력 형식과 이야기 설정을 빼고, 어떤 자료구조·알고리즘으로 무엇을 계산하는지 한 문장으로 적는다.
- JSON 외의 텍스트는 출력하지 않는다.

[출력 형식]
{{
  "difficulty": "{difficulty}",
  "category": "카테고리",
  "category_select_reason": "한 문장",
  "algorithm_core": "핵심 풀이 아이디어 한 문장",
  "problem_title": "문제 제목",
  "problem_description": "문제 지문",
  "input_format": "입력 형식 설명",
  "output_format": "출력 형식 설명",
  "problem_examples": [
    {{"input": "입력", "output": "출력", "explanation": "설명"}}
  ],
  "input_constraints": [
    {{"target": "N", "scope": "input", "data_type": "int",
      "min_value": "1", "max_value": "100000", "data_count": 1,
      "special_conditions": []}}
  ],
  "execution_limits": [
    {{"language": "python", "time_limit_ms": 2000, "memory_limit_kb": 262144}}
  ]
}}
"""

RESPONSE_CATEGORIES = [c.value for c in Category if c != Category.RANDOM]


async def fetch_few_shot(difficulty: Difficulty, category: str, k: int) -> list[dict]:
    """
    같은 난이도·카테고리의 기존 문제를 조회한다.

    문제 저장소가 준비되기 전까지는 빈 목록을 반환한다.

    Parameters:
        difficulty (Difficulty): 요청 난이도
        category (str): 요청 카테고리
        k (int): 조회할 문제 수

    Returns:
        list[dict]: 참고 문제 목록
    """
    return []


def render_few_shot(problems: list[dict]) -> str:
    """참고 문제 목록을 프롬프트용 텍스트로 만든다."""
    if not problems:
        return "(참고 문제 없음)"
    blocks = []
    for idx, problem in enumerate(problems, start=1):
        blocks.append(
            f"[예시 {idx}]\n"
            f"제목: {problem.get('problem_title')}\n"
            f"지문: {problem.get('problem_description')}\n"
            f"입력 형식: {problem.get('input_format')}\n"
            f"출력 형식: {problem.get('output_format')}"
        )
    return "\n\n".join(blocks)


def build_prompt(state: GraphState, few_shot: list[dict]) -> str:
    """
    문제 생성 프롬프트를 만든다.

    요청 카테고리가 RANDOM이면 응답 카테고리 목록 중 하나를 고르게 한다.

    Parameters:
        state (GraphState): 요청 난이도·카테고리가 담긴 상태
        few_shot (list[dict]): 참고 문제 목록

    Returns:
        str: 완성된 프롬프트
    """
    if state.requested_category == Category.RANDOM:
        category_instruction = "다음 중 하나를 골라 category에 적는다: " + ", ".join(
            RESPONSE_CATEGORIES
        )
    else:
        category_instruction = f"{state.requested_category} (category에 그대로 적는다)"

    return GENERATE_PROBLEM_PROMPT.format(
        difficulty=state.requested_difficulty.value,
        category_instruction=category_instruction,
        few_shot=render_few_shot(few_shot),
        languages=", ".join(lang.value for lang in Language),
    )


async def generate_problem(state: GraphState) -> dict:
    """
    요청 난이도·카테고리로 문제 지문, 제약, 공개 예시, 핵심 풀이 아이디어를 생성한다.

    Parameters:
        state (GraphState): 요청 난이도·카테고리가 담긴 상태

    Returns:
        dict: 생성한 문제 필드와 호출 설정

    Raises:
        LLMOutputParseError: 응답이 기대한 형식이 아닌 경우
    """
    cfg = LLMConfig(
        model_name=MODEL_NAME,
        prompt_version=PROMPT_VERSION,
        temperature=0.9,
        top_k=3,
    )

    few_shot = await fetch_few_shot(
        state.requested_difficulty, state.requested_category, cfg.top_k
    )
    prompt = build_prompt(state, few_shot)
    data = parse_json(await call_llm(prompt, cfg))

    try:
        return {
            "difficulty": Difficulty(data["difficulty"]),
            "category": data["category"],
            "category_select_reason": data["category_select_reason"],
            "algorithm_core": data["algorithm_core"],
            "problem_title": data["problem_title"],
            "problem_description": data["problem_description"],
            "input_format": data["input_format"],
            "output_format": data["output_format"],
            "problem_examples": [Example(**e) for e in data["problem_examples"]],
            "input_constraints": [
                InputConstraint(**c) for c in data["input_constraints"]
            ],
            "execution_limits": [
                ExecutionLimit(**lim) for lim in data["execution_limits"]
            ],
            "node_models": {"generate_problem": cfg},
        }
    except (KeyError, TypeError, ValueError) as error:
        raise LLMOutputParseError(f"문제 생성 응답 형식 오류: {error}") from error
