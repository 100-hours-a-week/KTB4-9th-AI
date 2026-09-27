from pydantic import BaseModel, Field

from src.client.llm import LLMConfig, call_llm_structured
from src.schema.evaluation import EvaluationRequest, Keyword

MODEL_NAME = "gemini-3.7-flash"
PROMPT_VERSION = "evaluate_nl_solution/v1"

MIN_SCORE = 0
MAX_SCORE = 100

EVALUATE_PROMPT = """당신은 코딩 테스트 자연어 풀이 채점자다.

[문제]
제목: {problem_title}
지문:
{problem_content}

[정답 카테고리]
{category} ({category_select_reason})

[핵심 키워드]
{solution_keywords}

[유저 풀이]
{natural_solution}

[채점 기준]
- 접근 방식이 문제를 올바르게 푸는가 (40점)
- 정답 카테고리의 접근을 사용했는가 (20점)
  · 정답 카테고리로 풀었으면 20점을 준다.
  · 다른 알고리즘이어도 정확하고 제약 안에서 동작하면 10점 이상 준다.
  · 정답보다 뚜렷하게 비효율적인 접근이면 5점 이하로 준다.
- 핵심 키워드의 내용이 풀이에 담겼는가 (30점)
- 설명이 실행 가능한 순서로 이어지는가 (10점)
- 단어만 나열하고 근거가 없으면 키워드 점수를 주지 않는다.
- 0 이상 100 이하의 정수로 채점한다.

[키워드 판정]
- 핵심 키워드마다 풀이에 그 내용이 담겼는지 판정한다.
- 단어가 그대로 나오지 않아도 같은 의미면 포함으로 본다.
  ("해시맵"을 "딕셔너리로 개수를 센다"로 설명했으면 포함)
- 단어만 나오고 어떻게 쓰는지 설명이 없으면 포함으로 보지 않는다.
- 주어진 키워드를 모두 판정하고, 없는 키워드를 만들지 않는다.

[피드백]
- 두 문장 이내로 쓴다. 맞은 부분과 빠진 부분을 각각 짚는다.
- 정답 코드를 알려주지 않는다.
"""


class KeywordVerdict(BaseModel):
    """키워드 하나의 포함 여부 판정."""

    keyword: str
    is_included: bool


class Evaluation(BaseModel):
    """자연어 풀이 채점 응답 형식."""

    score: int = Field(ge=MIN_SCORE, le=MAX_SCORE)
    feedback: str
    keywords: list[KeywordVerdict]


def render_keywords(keywords: list[str] | None) -> str:
    """핵심 키워드 목록을 프롬프트용 텍스트로 만든다."""
    if not keywords:
        return "(없음)"
    return "\n".join(f"- {keyword}" for keyword in keywords)


def build_prompt(request: EvaluationRequest) -> str:
    """
    자연어 풀이 채점 프롬프트를 만든다.

    Parameters:
        request (EvaluationRequest): 문제 정보와 유저 풀이

    Returns:
        str: 완성된 프롬프트
    """
    return EVALUATE_PROMPT.format(
        problem_title=request.problem_title or "(없음)",
        problem_content=request.problem_content or "(없음)",
        category=request.category or "(없음)",
        category_select_reason=request.category_select_reason or "(없음)",
        solution_keywords=render_keywords(request.solution_keywords),
        natural_solution=request.natural_solution or "(없음)",
    )


def align_keywords(
    requested: list[str] | None, verdicts: list[KeywordVerdict]
) -> list[Keyword]:
    """
    판정 결과를 요청한 키워드 목록에 맞춘다.

    모델이 키워드를 빠뜨리거나 없는 키워드를 만들어도
    요청한 키워드만 요청한 순서로 반환한다.

    Parameters:
        requested (list[str] | None): 요청에 담긴 핵심 키워드
        verdicts (list[KeywordVerdict]): 모델의 판정 결과

    Returns:
        list[Keyword]: 요청한 키워드와 포함 여부
    """
    included = {
        verdict.keyword.strip(): verdict.is_included
        for verdict in verdicts
        if verdict.keyword.strip()
    }
    return [
        Keyword(keyword=keyword, is_included=included.get(keyword, False))
        for keyword in (requested or [])
    ]


async def evaluate(request: EvaluationRequest) -> Evaluation:
    """
    유저의 자연어 풀이를 채점한다.

    Parameters:
        request (EvaluationRequest): 문제 정보와 유저 풀이

    Returns:
        Evaluation: 점수, 피드백, 키워드별 판정

    Raises:
        LLMOutputParseError: 응답이 형식에 맞지 않는 경우
    """
    cfg = LLMConfig(
        model_name=MODEL_NAME,
        prompt_version=PROMPT_VERSION,
        temperature=0.0,
    )
    return await call_llm_structured(build_prompt(request), cfg, Evaluation)
