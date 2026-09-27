import asyncio
import logging
import random

from pydantic import BaseModel, Field

from src.client.embedding import EMBEDDING_MODEL, embed_text
from src.client.judge0 import run_code
from src.client.llm import LLMConfig, call_llm_structured
from src.core.enums import Category, DiscardReason
from src.core.exception import LLMOutputParseError, ProblemSaveError
from src.db.repository import (
    BattleProblemEmbeddingRepository,
    BattleProblemRepository,
)
from src.db.session import session_scope
from src.problem.battle.state import (
    MAX_CONTENT_LENGTH,
    MAX_TITLE_LENGTH,
    REFERENCE_LANGUAGE,
    REFERENCE_LIMIT,
    TEST_CASE_COUNT,
    BattleState,
    BattleTestCase,
    discard,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-3.7-flash"
PROMPT_VERSION = "generate_battle_problem/v1"

# 채점 서버(judge0 language_id 71)의 파이썬 버전.
REFERENCE_RUNTIME = "Python 3.8"

# 중복 판정 기준. 일반 문제와 같은 값을 쓴다.
DUPLICATE_SIMILARITY = 0.85
NEIGHBOR_COUNT = 3

RESPONSE_CATEGORIES = [c for c in Category if c != Category.RANDOM]

GENERATE_PROMPT = """당신은 코딩 테스트 배틀 문제 출제자다.

[요청]
카테고리: {category}

[규칙]
- 짧은 시간 안에 손으로 풀 수 있는 쉬운 문제를 한국어로 만든다.
- problem_title은 {max_title}자 이하로 쓴다.
- problem_content는 {max_content}자 이하로 쓰고,
  입력과 출력 형식을 지문 안에 함께 설명한다.
- test_inputs는 서로 다른 입력 {count}개다. 출력은 적지 않는다.
- algorithm_core에는 이야기 설정을 빼고, 어떤 자료구조·알고리즘으로
  무엇을 계산하는지 한 문장으로 적는다.
- reference_code는 이 문제를 푸는 {language} 코드다.
  표준 입력으로 읽고 표준 출력으로 쓴다. 표준 라이브러리만 사용한다.
  {runtime}에서 실행되므로 이후 버전에서 추가된 문법과 함수는 쓰지 않는다.
  설명과 주석 없이 실행 가능한 소스 코드만 넣는다.
"""


class GeneratedBattleProblem(BaseModel):
    """배틀 문제 생성 응답 형식."""

    problem_title: str
    problem_content: str
    algorithm_core: str = Field(description="이야기 설정을 뺀 핵심 풀이 한 문장")
    test_inputs: list[str] = Field(description="서로 다른 테스트 입력")
    reference_code: str


def build_prompt(category: Category) -> str:
    """배틀 문제 생성 프롬프트를 만든다."""
    return GENERATE_PROMPT.format(
        category=category.value,
        max_title=MAX_TITLE_LENGTH,
        max_content=MAX_CONTENT_LENGTH,
        count=TEST_CASE_COUNT,
        language=REFERENCE_LANGUAGE.value,
        runtime=REFERENCE_RUNTIME,
    )


async def generate_battle_problem(state: BattleState) -> dict:
    """
    배틀 문제와 테스트 입력, 검증용 코드를 한 번에 생성한다.

    카테고리는 코드가 무작위로 정한다. 재시도로 다시 들어오면 같은
    카테고리를 유지하고 문제만 새로 만든다.

    Parameters:
        state (BattleState): 현재 상태

    Returns:
        dict: 생성한 문제 필드와 호출 설정. 실패하면 폐기 정보
    """
    cfg = LLMConfig(
        model_name=MODEL_NAME, prompt_version=PROMPT_VERSION, temperature=0.9
    )
    category = state.category or random.choice(RESPONSE_CATEGORIES)

    try:
        problem = await call_llm_structured(
            build_prompt(category), cfg, GeneratedBattleProblem
        )
    except LLMOutputParseError as error:
        return discard(
            DiscardReason.LLM_ERROR,
            "generate_battle_problem",
            f"배틀 문제 생성 실패: {error}",
        )

    return {
        "category": category,
        "problem_title": problem.problem_title,
        "problem_content": problem.problem_content,
        "algorithm_core": problem.algorithm_core,
        "test_cases": [BattleTestCase(input=i) for i in problem.test_inputs],
        "reference_code": problem.reference_code,
        "node_models": {"generate_battle_problem": cfg},
    }


async def static_validate(state: BattleState) -> dict:
    """
    LLM 없이 규칙만으로 생성 결과를 검사한다.

    Parameters:
        state (BattleState): 생성 결과가 채워진 상태

    Returns:
        dict: 통과하면 빈 dict, 실패하면 폐기 정보
    """
    stage = "static_validate"

    required = {
        "카테고리": state.category,
        "제목": state.problem_title,
        "지문": state.problem_content,
        "핵심 풀이": state.algorithm_core,
        "검증용 코드": state.reference_code,
    }
    for name, value in required.items():
        if not value:
            return discard(DiscardReason.EMPTY_FIELD, stage, f"{name}이(가) 비었음")

    if len(state.problem_title) > MAX_TITLE_LENGTH:
        return discard(
            DiscardReason.VALIDATION_FAILED,
            stage,
            f"제목이 {MAX_TITLE_LENGTH}자를 넘음: {len(state.problem_title)}자",
        )

    if len(state.problem_content) > MAX_CONTENT_LENGTH:
        return discard(
            DiscardReason.VALIDATION_FAILED,
            stage,
            f"지문이 {MAX_CONTENT_LENGTH}자를 넘음: {len(state.problem_content)}자",
        )

    inputs = {case.input.strip() for case in state.test_cases if case.input.strip()}
    if len(inputs) != TEST_CASE_COUNT:
        return discard(
            DiscardReason.VALIDATION_FAILED,
            stage,
            f"서로 다른 테스트 입력이 {TEST_CASE_COUNT}개가 아님: {len(inputs)}개",
        )

    return {}


async def fill_outputs(state: BattleState) -> dict:
    """
    검증용 코드를 각 입력으로 실행해 기대 출력을 채운다.

    모델이 출력을 직접 적으면 틀려도 확인할 방법이 없으므로 실행 결과로 채운다.
    하나라도 실패하면 문제부터 다시 만든다.

    Parameters:
        state (BattleState): 정적 검증을 통과한 상태

    Returns:
        dict: 출력이 채워진 테스트케이스. 실패하면 재시도 또는 폐기 정보
    """
    stage = "fill_outputs"

    if state.fill_attempt >= state.max_fill_attempt:
        return discard(
            DiscardReason.MAX_ATTEMPT_EXCEEDED,
            stage,
            f"검증용 코드 실행이 {state.max_fill_attempt}회 모두 실패",
        )

    results = await asyncio.gather(
        *(
            run_code(
                state.reference_code, REFERENCE_LANGUAGE, case.input, REFERENCE_LIMIT
            )
            for case in state.test_cases
        )
    )

    filled: list[BattleTestCase] = []
    for case, result in zip(state.test_cases, results, strict=True):
        output = result.stdout.strip()
        if not result.succeeded or not output:
            logger.info(
                "배틀 검증용 코드 실행 실패: 입력 %r → %s (%s)",
                case.input,
                result.status,
                result.stderr[:100],
            )
            return {"fill_attempt": state.fill_attempt + 1}
        filled.append(BattleTestCase(input=case.input, output=output))

    return {"test_cases": filled}


async def check_duplicate(state: BattleState) -> dict:
    """
    같은 카테고리의 기존 배틀 문제와 비슷한지 본다.

    일반 문제와 데일리 문제는 비교 대상이 아니다. 배틀끼리만 본다.
    임베딩은 상태에 담아 finalize가 색인에 다시 쓴다.

    Parameters:
        state (BattleState): 출력까지 채워진 상태

    Returns:
        dict: 신규면 임베딩. 중복이거나 실패하면 폐기 정보
    """
    stage = "check_duplicate"

    try:
        embedding = await embed_text(state.algorithm_core)
        async with session_scope() as session:
            neighbors = await BattleProblemEmbeddingRepository(session).find_similar(
                category=state.category,
                embedding=embedding,
                embedding_model=EMBEDDING_MODEL,
                limit=NEIGHBOR_COUNT,
            )
    except Exception as error:
        # 임베딩 호출 실패, DB 장애 모두 여기로 온다.
        # 확인하지 못한 문제를 저장하지 않는 쪽을 택한다.
        logger.warning("배틀 중복 검사 실패: %s", error, exc_info=True)
        return discard(DiscardReason.LLM_ERROR, stage, f"배틀 중복 검사 실패: {error}")

    if not neighbors:
        return {"algorithm_core_embedding": embedding}

    closest = neighbors[0]
    if closest.similarity >= DUPLICATE_SIMILARITY:
        return discard(
            DiscardReason.DUPLICATE,
            stage,
            f"유사도 {closest.similarity:.3f} (기준 {DUPLICATE_SIMILARITY}) "
            f"- 기존 배틀 문제 {closest.problem_id}: {closest.algorithm_core}",
        )

    return {"algorithm_core_embedding": embedding}


async def finalize(state: BattleState) -> dict:
    """
    확정된 배틀 문제를 저장한다.

    실패하면 예외를 올린다. 만들어 낸 문제가 사라지는 것이므로
    성공으로 위장해선 안 된다.

    Parameters:
        state (BattleState): 검증을 모두 통과한 상태

    Returns:
        dict: 저장된 문제의 id

    Raises:
        ProblemSaveError: 저장에 실패한 경우
    """
    try:
        async with session_scope() as session:
            row = await BattleProblemRepository(session).add(
                category=state.category,
                problem_title=state.problem_title,
                problem_content=state.problem_content,
                test_cases=[case.model_dump(mode="json") for case in state.test_cases],
            )
            problem_id = row.id

            if state.algorithm_core_embedding:
                await BattleProblemEmbeddingRepository(session).add(
                    battle_problem_id=problem_id,
                    category=state.category,
                    algorithm_core=state.algorithm_core,
                    embedding=state.algorithm_core_embedding,
                    embedding_model=EMBEDDING_MODEL,
                )
    except Exception as error:
        logger.error("배틀 문제 저장 실패: %s", error, exc_info=True)
        raise ProblemSaveError(f"배틀 문제 저장 실패: {error}") from error

    logger.info(
        "배틀 문제 저장: %s / %s / %s",
        problem_id,
        state.category.value,
        state.problem_title,
    )
    return {}
