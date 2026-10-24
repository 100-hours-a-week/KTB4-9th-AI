import asyncio

from pydantic import BaseModel, Field

from src.client.llm import LLMConfig, call_llm_structured
from src.core.config import get_settings
from src.schema.evaluation import EvaluationRequest, Keyword

PROMPT_VERSION = "evaluate_nl_solution/v2"

# 항목별 배점. 합이 100이다.
APPROACH_MAX = 40
CATEGORY_MAX = 20
KEYWORD_MAX = 30
ORDER_MAX = 10

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
- 접근 방식이 문제를 올바르게 푸는가 ({approach_max}점)
- 정답 카테고리의 접근을 사용했는가 ({category_max}점)
  · 정답 카테고리로 풀었으면 만점을 준다.
  · 다른 알고리즘이어도 정확하고 제약 안에서 동작하면 절반 이상 준다.
  · 정답보다 뚜렷하게 비효율적인 접근이면 4분의 1 이하로 준다.
- 설명이 실행 가능한 순서로 이어지는가 ({order_max}점)
- 세 항목을 각각 매긴다. 합계는 적지 않는다.
- 핵심 키워드 점수는 매기지 않는다. 포함 여부만 판정한다.

[0점 규칙]
- 풀이에 구체적인 방법이 없으면 접근 방식과 카테고리를 모두 0점으로 한다.
  "잘 풉니다", "쉽게 풀 수 있어요", "알고리즘으로 풉니다"처럼
  의도나 자신감만 적은 경우가 여기 해당한다.
  어떤 자료구조로 무엇을 어떤 순서로 처리하는지가 없으면 0점이다.
- 의도, 태도, 자신감은 점수에 반영하지 않는다.
- 풀이가 비어 있으면 세 항목 모두 0점으로 한다.
- 다만 방법이 적혀 있으면 짧아도 그 내용으로 판단한다.
  길이를 이유로 점수를 깎지 않는다.

[채점 예시]
풀이: "잘 풉니다"
→ 접근 0, 카테고리 0, 순서 0. 키워드는 모두 미포함.
  구체적인 방법이 없어 판단할 근거가 없다.

[키워드 판정]
- 핵심 키워드마다 풀이에 그 내용이 담겼는지 판정한다.
- 단어가 그대로 나오지 않아도 같은 의미면 포함으로 본다.
  ("해시맵"을 "딕셔너리로 개수를 센다"로 설명했으면 포함)
- 단어만 나오고 어떻게 쓰는지 설명이 없으면 포함으로 보지 않는다.
- 주어진 키워드를 모두 판정하고, 없는 키워드를 만들지 않는다.

[피드백]
- 두 문장 이내로 쓴다. 맞은 부분과 빠진 부분을 각각 짚는다.
- 풀이에 내용이 없으면 칭찬하지 않는다. 무엇을 적어야 하는지만 알려준다.
- 정답 코드를 알려주지 않는다.
"""


class KeywordVerdict(BaseModel):
    """키워드 하나의 포함 여부 판정."""

    keyword: str
    is_included: bool


class Evaluation(BaseModel):
    """자연어 풀이 채점 응답 형식. 항목별로 받고 합산은 코드가 한다."""

    approach_score: int = Field(ge=0, le=APPROACH_MAX)
    category_score: int = Field(ge=0, le=CATEGORY_MAX)
    order_score: int = Field(ge=0, le=ORDER_MAX)
    feedback: str
    keywords: list[KeywordVerdict]


class ScoredEvaluation(BaseModel):
    """합산까지 끝난 채점 결과."""

    score: int
    approach_score: int
    category_score: int
    keyword_score: int
    order_score: int
    feedback: str
    keywords: list[Keyword]


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
        approach_max=APPROACH_MAX,
        category_max=CATEGORY_MAX,
        order_max=ORDER_MAX,
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


def keyword_score(keywords: list[Keyword]) -> int:
    """
    포함 판정 결과로 키워드 점수를 계산한다.

    모델은 포함 여부만 판정하고 점수는 매기지 않는다.

    Parameters:
        keywords (list[Keyword]): 요청한 키워드와 포함 여부

    Returns:
        int: 0 이상 KEYWORD_MAX 이하의 점수
    """
    if not keywords:
        return 0
    included = sum(1 for keyword in keywords if keyword.is_included)
    return round(KEYWORD_MAX * included / len(keywords))


def score_evaluation(
    evaluation: Evaluation, requested: list[str] | None
) -> ScoredEvaluation:
    """
    모델 응답에 키워드 점수를 더해 최종 결과를 만든다.

    Parameters:
        evaluation (Evaluation): 모델이 매긴 항목별 점수와 판정
        requested (list[str] | None): 요청에 담긴 핵심 키워드

    Returns:
        ScoredEvaluation: 총점까지 계산된 결과
    """
    keywords = align_keywords(requested, evaluation.keywords)
    keyword = keyword_score(keywords)
    return ScoredEvaluation(
        score=(
            evaluation.approach_score
            + evaluation.category_score
            + keyword
            + evaluation.order_score
        ),
        approach_score=evaluation.approach_score,
        category_score=evaluation.category_score,
        keyword_score=keyword,
        order_score=evaluation.order_score,
        feedback=evaluation.feedback,
        keywords=keywords,
    )


async def evaluate(request: EvaluationRequest) -> ScoredEvaluation:
    """
    유저의 자연어 풀이를 채점한다.

    Parameters:
        request (EvaluationRequest): 문제 정보와 유저 풀이

    Returns:
        ScoredEvaluation: 총점, 항목별 점수, 피드백, 키워드별 판정

    Raises:
        LLMOutputParseError: 응답이 형식에 맞지 않는 경우
    """
    settings = get_settings()
    if settings.mock_llm:
        await asyncio.sleep(1.5)
        return score_evaluation(
            Evaluation(
                approach_score=35,
                category_score=20,
                order_score=10,
                feedback=(
                    "[MOCK] 핵심 알고리즘 접근 방식이 올바르며 요구사항을 만족합니다."
                ),
                keywords=[
                    KeywordVerdict(keyword=k, is_included=True)
                    for k in (request.solution_keywords or [])
                ],
            ),
            request.solution_keywords,
        )

    cfg = LLMConfig(
        model_name=get_settings().llm_models.evaluation_nl_solution,
        prompt_version=PROMPT_VERSION,
        temperature=0.0,
    )
    evaluation = await call_llm_structured(build_prompt(request), cfg, Evaluation)
    return score_evaluation(evaluation, request.solution_keywords)
