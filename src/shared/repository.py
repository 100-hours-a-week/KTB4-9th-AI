import uuid
from collections.abc import Sequence
from typing import Any, NamedTuple

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.enums import Category, Difficulty, DiscardReason
from src.schema import FewshotSeedCreate, Problem
from src.shared.model import (
    DiscardedProblem,
    FewshotSeed,
    GeneratedProblem,
    ProblemEmbedding,
)


class SimilarProblem(NamedTuple):
    problem_id: uuid.UUID
    algorithm_core: str
    similarity: float  # 코사인 유사도 (1에 가까울수록 비슷함)


# ── 1. 생성 버퍼 ───────────────────────────────────────────────────────


class GeneratedProblemRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, problem: Problem) -> GeneratedProblem:
        row = GeneratedProblem(**problem.model_dump(mode="json"))
        self.session.add(row)
        await self.session.flush()  # id 확정 (임베딩 저장에 필요)
        return row

    async def get(self, problem_id: uuid.UUID) -> GeneratedProblem | None:
        return await self.session.get(GeneratedProblem, problem_id)

    async def list_pending(
        self,
        category: Category | None = None,
        difficulty: Difficulty | None = None,
        limit: int = 50,
    ) -> Sequence[GeneratedProblem]:
        """미전송 문제를 오래된 순으로. 조건을 비우면 전체.

        다른 프로세스가 같은 행을 동시에 보내지 않도록 행 잠금을 건다.
        잠금은 트랜잭션이 끝날 때 풀리므로 전송 → mark_sent를 같은 세션에서 한다.
        """
        stmt = select(GeneratedProblem).where(GeneratedProblem.sent_at.is_(None))
        if category is not None:
            stmt = stmt.where(GeneratedProblem.category == category)
        if difficulty is not None:
            stmt = stmt.where(GeneratedProblem.difficulty == difficulty)
        stmt = (
            stmt.order_by(GeneratedProblem.created_at)
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        return (await self.session.scalars(stmt)).all()

    async def mark_sent(self, problem_ids: Sequence[uuid.UUID]) -> int:
        """전송 완료 표시. 실제로 바뀐 행 수를 돌려준다."""
        if not problem_ids:
            return 0
        stmt = (
            update(GeneratedProblem)
            .where(
                GeneratedProblem.id.in_(problem_ids),
                GeneratedProblem.sent_at.is_(None),
            )
            .values(sent_at=func.now())
        )
        result = await self.session.execute(stmt)
        return result.rowcount

    @staticmethod
    def to_problem(row: GeneratedProblem) -> Problem:
        return Problem.model_validate(row, from_attributes=True)


# ── 2. 폐기 로그 ───────────────────────────────────────────────────────


class DiscardedProblemRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(
        self,
        *,
        requested_category: Category,
        requested_difficulty: Difficulty,
        stage: str,
        reason: DiscardReason,
        detail: str | None = None,
        raw_payload: dict[str, Any] | None = None,
    ) -> DiscardedProblem:
        row = DiscardedProblem(
            requested_category=requested_category,
            requested_difficulty=requested_difficulty,
            stage=stage,
            discard_reason=reason,
            discard_detail=detail,
            raw_payload=raw_payload,
        )
        self.session.add(row)
        return row


# ── 3. 임베딩 ──────────────────────────────────────────────────────────


class ProblemEmbeddingRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(
        self,
        *,
        problem_id: uuid.UUID,
        category: Category,
        algorithm_core: str,
        embedding: list[float],
        embedding_model: str,
    ) -> ProblemEmbedding:
        row = ProblemEmbedding(
            problem_id=problem_id,
            category=category,
            algorithm_core=algorithm_core,
            embedding=embedding,
            embedding_model=embedding_model,
        )
        self.session.add(row)
        return row

    async def find_similar(
        self,
        *,
        category: Category,
        embedding: list[float],
        embedding_model: str,
        limit: int = 5,
    ) -> list[SimilarProblem]:
        """같은 카테고리 + 같은 임베딩 모델 안에서 가장 비슷한 순으로."""
        distance = ProblemEmbedding.embedding.cosine_distance(embedding)
        stmt = (
            select(
                ProblemEmbedding.problem_id,
                ProblemEmbedding.algorithm_core,
                (1 - distance).label("similarity"),
            )
            .where(
                ProblemEmbedding.category == category,
                ProblemEmbedding.embedding_model == embedding_model,
            )
            .order_by(distance)
            .limit(limit)
        )
        rows = await self.session.execute(stmt)
        return [SimilarProblem(*r) for r in rows]


# ── 4. few-shot 시드 ───────────────────────────────────────────────────


class FewshotSeedRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, seed: FewshotSeedCreate) -> FewshotSeed:
        row = FewshotSeed(**seed.model_dump(mode="json"))
        self.session.add(row)
        await self.session.flush()
        return row

    async def sample(
        self, category: Category, difficulty: Difficulty, k: int = 3
    ) -> Sequence[FewshotSeed]:
        """프롬프트용. 해당 카테고리·난이도의 활성 시드 중 무작위 k개.

        category에 RANDOM을 넘기지 말 것. 서비스에서 실제 카테고리로 바꾼 뒤 호출한다.
        """
        stmt = (
            select(FewshotSeed)
            .where(
                FewshotSeed.category == category,
                FewshotSeed.difficulty == difficulty,
                FewshotSeed.is_active.is_(True),
            )
            .order_by(func.random())
            .limit(k)
        )
        return (await self.session.scalars(stmt)).all()
