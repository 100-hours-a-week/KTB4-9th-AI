"""임베딩 생성.

중복 검사용이다. 지문 전체가 아니라 algorithm_core만 임베딩한다.
이야기 설정과 입출력 형식이 달라도 같은 알고리즘이면 중복으로 잡아야 한다.

모델은 3072차원이 기본이지만 output_dimensionality로 줄일 수 있고,
줄인 결과도 노름 1의 단위 벡터로 나온다. DB 컬럼(Vector(EMBEDDING_DIM))에
맞춰 요청하므로 차원이 어긋날 일은 없다.

색인할 때와 조회할 때 task_type이 같아야 벡터를 비교할 수 있다.
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.core.config import get_settings
from src.shared.model import EMBEDDING_DIM

EMBEDDING_MODEL = "models/gemini-embedding-2"

# 검색이 아니라 두 문장이 같은 뜻인지 보는 용도
TASK_TYPE = "SEMANTIC_SIMILARITY"


def build_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    임베딩 모델 객체를 만든다.

    Returns:
        GoogleGenerativeAIEmbeddings: 호출 가능한 임베딩 객체

    Raises:
        RuntimeError: API 키가 설정되지 않은 경우
    """
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=settings.gemini_api_key,
        output_dimensionality=EMBEDDING_DIM,
        task_type=TASK_TYPE,
    )


async def embed_text(text: str) -> list[float]:
    """
    문장 하나를 벡터로 만든다.

    Parameters:
        text (str): 임베딩할 문장

    Returns:
        list[float]: EMBEDDING_DIM 길이의 벡터

    Raises:
        RuntimeError: API 키가 없거나 반환된 차원이 컬럼과 다른 경우
    """
    vector = await build_embeddings().aembed_query(text)
    if len(vector) != EMBEDDING_DIM:
        raise RuntimeError(f"임베딩 차원이 {EMBEDDING_DIM}이 아님: {len(vector)}")
    return vector
