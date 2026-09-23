import logging

from pydantic import BaseModel, Field

from src.enum import DiscardReason
from src.problem.render import render_problem
from src.problem.state import GraphState, LLMConfig, discard
from src.shared.llm import call_llm_structured

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-3.5-flash-lite"
PROMPT_VERSION = "generate_nl_keyword/v1"

KEYWORD_MIN = 3
KEYWORD_MAX = 7

GENERATE_NL_KEYWORD_PROMPT = """당신은 코딩 테스트 채점자다.

[문제]
{problem}

[핵심 풀이]
{algorithm_core}

[규칙]
- 학습자가 이 문제의 풀이를 자연어로 설명할 때
  반드시 언급해야 하는 핵심 키워드를 {minimum}~{maximum}개 뽑는다.
- 자료구조, 알고리즘, 풀이의 핵심 아이디어를 가리키는 말만 넣는다.
- 문제의 소재(등장인물, 사물 이름)와 입출력 형식은 넣지 않는다.
- 한 키워드에 한 개념만 담는다. 설명하는 문장을 넣지 않는다.
- 코드 문법, 함수 이름, 변수 이름은 넣지 않는다.
- 같은 개념을 다른 말로 반복하지 않는다.
"""


class SolutionKeywords(BaseModel):
    """자연어 채점 키워드 응답 형식."""

    keywords: list[str] = Field(
        description="자연어 풀이에 반드시 나와야 하는 핵심 개념"
    )


def build_prompt(state: GraphState) -> str:
    """
    채점 키워드 추출 프롬프트를 만든다.

    카테고리와 선정 이유를 함께 넣는다. 채점자는 문제의 분류를 아는 상태이고,
    분류 자체가 기대 키워드의 단서가 된다.

    Parameters:
        state (GraphState): 문제 필드와 algorithm_core가 채워진 상태

    Returns:
        str: 완성된 프롬프트
    """
    return GENERATE_NL_KEYWORD_PROMPT.format(
        problem=render_problem(state, include_category=True),
        algorithm_core=state.algorithm_core or "(없음)",
        minimum=KEYWORD_MIN,
        maximum=KEYWORD_MAX,
    )


def normalize_keywords(keywords: list[str]) -> list[str]:
    """
    앞뒤 공백과 빈 값을 걷어내고 중복을 없앤다.

    대소문자만 다른 값도 같은 키워드로 본다. 순서는 모델이 준 순서를 지킨다.

    Parameters:
        keywords (list[str]): 모델이 준 키워드 목록

    Returns:
        list[str]: 정리한 키워드 목록. KEYWORD_MAX개까지만
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        text = keyword.strip()
        if not text or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        cleaned.append(text)
    return cleaned[:KEYWORD_MAX]


async def generate_nl_keyword(state: GraphState) -> dict:
    """
    자연어 풀이 채점에 쓸 핵심 키워드를 뽑는다.

    병렬 노드이므로 예외를 올리지 않는다. 올리면 그래프 전체가 죽는다.

    Parameters:
        state (GraphState): 의미 검증을 통과한 상태

    Returns:
        dict: 채점 키워드와 호출 설정. 실패 시 폐기 정보
    """
    cfg = LLMConfig(
        model_name=MODEL_NAME,
        prompt_version=PROMPT_VERSION,
        temperature=0.2,
    )

    try:
        response = await call_llm_structured(build_prompt(state), cfg, SolutionKeywords)
    except Exception as error:
        # 응답 형식 오류뿐 아니라 키 누락, 타임아웃, 레이트 리밋도 여기로 온다.
        # 병렬 노드에서 예외를 올리면 그래프 전체가 죽으므로 폐기로 바꾼다.
        logger.warning("채점 키워드 생성 실패: %s", error, exc_info=True)
        return discard(DiscardReason.LLM_ERROR, f"채점 키워드 생성 실패: {error}")

    keywords = normalize_keywords(response.keywords)
    if len(keywords) < KEYWORD_MIN:
        return discard(
            DiscardReason.LLM_ERROR,
            f"채점 키워드가 {KEYWORD_MIN}개 미만: {len(keywords)}개",
        )

    return {
        "solution_keywords": keywords,
        "node_models": {"generate_nl_keyword": cfg},
    }
