import contextlib
import uuid

import pytest

from src.core.constants import EMBEDDING_DIM
from src.core.enums import Category, DiscardReason
from src.db.repository import SimilarProblem
from src.problem.nodes import check_duplicate as module
from src.problem.nodes.check_duplicate import (
    DUPLICATE_SIMILARITY,
    NEIGHBOR_COUNT,
    check_duplicate,
)
from src.problem.state import GraphState
from tests import fake_nodes

ALGORITHM_CORE = "해시맵으로 M - x의 등장 횟수를 누적해 합이 M인 쌍의 수를 센다"
VECTOR = [0.1] * EMBEDDING_DIM


async def make_state(**updates) -> GraphState:
    """정적 검증을 통과한 상태. 카테고리는 이미 정해져 있다."""
    base = GraphState(requested_difficulty="LV2", requested_category="HASH_TABLE")
    state = base.model_copy(update=await fake_nodes.generate_problem(base))
    return state.model_copy(
        update={
            "category": "HASH_TABLE",
            "algorithm_core": ALGORITHM_CORE,
            **updates,
        }
    )


def similar(similarity: float) -> SimilarProblem:
    return SimilarProblem(
        problem_id=uuid.uuid4(),
        algorithm_core="기존 문제의 핵심 풀이",
        similarity=similarity,
    )


def fix_lookup(
    monkeypatch: pytest.MonkeyPatch, neighbors: list[SimilarProblem]
) -> dict:
    """임베딩 호출과 DB 조회를 가짜로 바꿔 끼운다.

    Returns:
        dict: 임베딩에 넘어간 문장과 조회 인자가 담긴다
    """
    seen: dict = {}

    async def fake_embed(text: str) -> list[float]:
        seen["text"] = text
        return VECTOR

    @contextlib.asynccontextmanager
    async def fake_scope():
        yield object()

    class FakeRepository:
        def __init__(self, session) -> None:
            pass

        async def find_similar(self, **kwargs) -> list[SimilarProblem]:
            seen.update(kwargs)
            return neighbors

    monkeypatch.setattr(module, "embed_text", fake_embed)
    monkeypatch.setattr(module, "session_scope", fake_scope)
    monkeypatch.setattr(module, "ProblemEmbeddingRepository", FakeRepository)
    return seen


# ── 판정 ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_table_means_new(monkeypatch: pytest.MonkeyPatch) -> None:
    fix_lookup(monkeypatch, [])

    result = await check_duplicate(await make_state())

    assert result["is_duplicated"] is False
    assert "is_discarded" not in result


@pytest.mark.asyncio
async def test_similarity_at_the_threshold_is_a_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fix_lookup(monkeypatch, [similar(DUPLICATE_SIMILARITY)])

    result = await check_duplicate(await make_state())

    assert result["is_discarded"] is True
    assert result["discard_reason"] == DiscardReason.DUPLICATE


@pytest.mark.asyncio
async def test_similarity_just_below_the_threshold_is_new(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fix_lookup(monkeypatch, [similar(DUPLICATE_SIMILARITY - 0.01)])

    result = await check_duplicate(await make_state())

    assert result["is_duplicated"] is False


@pytest.mark.asyncio
async def test_the_closest_neighbor_decides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """가장 비슷한 하나로 판정한다. 나머지는 로그에만 남는다."""
    fix_lookup(
        monkeypatch,
        [similar(0.95), similar(0.5), similar(0.4)],
    )

    result = await check_duplicate(await make_state())

    assert result["is_discarded"] is True


@pytest.mark.asyncio
async def test_duplicate_detail_names_the_existing_problem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    neighbor = similar(0.97)
    fix_lookup(monkeypatch, [neighbor])

    result = await check_duplicate(await make_state())

    assert str(neighbor.problem_id) in result["discard_detail"]
    assert "0.970" in result["discard_detail"]


# ── 조회 방식 ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_only_the_algorithm_core_is_embedded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """지문 전체가 아니라 핵심 풀이만 임베딩한다."""
    seen = fix_lookup(monkeypatch, [])
    state = await make_state()

    await check_duplicate(state)

    assert seen["text"] == ALGORITHM_CORE
    assert state.problem_content not in seen["text"]


@pytest.mark.asyncio
async def test_comparison_is_limited_to_the_same_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = fix_lookup(monkeypatch, [])

    await check_duplicate(await make_state())

    assert seen["category"] is Category.HASH_TABLE
    assert seen["limit"] == NEIGHBOR_COUNT


@pytest.mark.asyncio
async def test_embedding_is_kept_for_finalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """finalize가 색인에 다시 쓰므로 임베딩을 두 번 부르지 않는다."""
    fix_lookup(monkeypatch, [])

    result = await check_duplicate(await make_state())

    assert result["algorithm_core_embedding"] == VECTOR


# ── 사전 조건과 실패 ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_algorithm_core_is_discarded() -> None:
    result = await check_duplicate(await make_state(algorithm_core=None))

    assert result["discard_reason"] == DiscardReason.EMPTY_FIELD


@pytest.mark.asyncio
async def test_missing_category_is_discarded() -> None:
    result = await check_duplicate(await make_state(category=None))

    assert result["discard_reason"] == DiscardReason.EMPTY_FIELD


@pytest.mark.asyncio
async def test_unknown_category_is_discarded() -> None:
    result = await check_duplicate(await make_state(category="없는카테고리"))

    assert result["discard_reason"] == DiscardReason.EMPTY_FIELD


@pytest.mark.asyncio
async def test_embedding_failure_is_discarded_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """확인하지 못한 문제를 저장하지 않는다."""

    async def broken(text: str) -> list[float]:
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")

    monkeypatch.setattr(module, "embed_text", broken)

    result = await check_duplicate(await make_state())

    assert result["is_discarded"] is True
    assert result["discard_reason"] == DiscardReason.LLM_ERROR


@pytest.mark.asyncio
async def test_database_failure_is_discarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_embed(text: str) -> list[float]:
        return VECTOR

    @contextlib.asynccontextmanager
    async def broken_scope():
        raise OSError("connection refused")
        yield  # pragma: no cover

    monkeypatch.setattr(module, "embed_text", fake_embed)
    monkeypatch.setattr(module, "session_scope", broken_scope)

    result = await check_duplicate(await make_state())

    assert result["discard_reason"] == DiscardReason.LLM_ERROR
