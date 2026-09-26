import uuid
from collections.abc import Collection, Sequence
from typing import Any, NamedTuple

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.enums import Category, Difficulty, DiscardReason, ProblemPurpose, Trigger
from src.db.models import (
    DiscardedProblem,
    FewshotSeed,
    GeneratedProblem,
    ProblemEmbedding,
)
from src.schema.problem import FewshotSeedCreate, Problem


class SimilarProblem(NamedTuple):
    problem_id: uuid.UUID
    algorithm_core: str
    similarity: float  # 코사인 유사도 (1에 가까울수록 비슷함)


# ── 1. 생성 버퍼 ───────────────────────────────────────────────────────


class GeneratedProblemRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(
        self,
        problem: Problem,
        *,
        trigger: Trigger,
        purpose: ProblemPurpose,
    ) -> GeneratedProblem:
        row = GeneratedProblem(
            **problem.model_dump(mode="json"), trigger=trigger, purpose=purpose
        )
        self.session.add(row)
        await self.session.flush()  # id 확정 (임베딩 저장에 필요)
        return row

    async def get(self, problem_id: uuid.UUID) -> GeneratedProblem | None:
        return await self.session.get(GeneratedProblem, problem_id)

    async def count_pending(
        self, purpose: ProblemPurpose = ProblemPurpose.NORMAL
    ) -> dict[tuple[Category, Difficulty], int]:
        """배치로 만들어 두고 아직 보내지 않은 재고를 조합별로 센다.

        생성 개수를 정할 때 Spring 재고와 합산한다. 없는 조합은 키가 없다.
        """
        stmt = (
            select(
                GeneratedProblem.category,
                GeneratedProblem.difficulty,
                func.count(),
            )
            .where(
                GeneratedProblem.sent_at.is_(None),
                GeneratedProblem.trigger == Trigger.BATCH,
                GeneratedProblem.purpose == purpose,
            )
            .group_by(GeneratedProblem.category, GeneratedProblem.difficulty)
        )
        rows = await self.session.execute(stmt)
        return {(category, difficulty): count for category, difficulty, count in rows}

    async def list_pending(
        self,
        purpose: ProblemPurpose,
        limit: int = 50,
        exclude_ids: Collection[uuid.UUID] = (),
    ) -> Sequence[GeneratedProblem]:
        """배치로 만든 미전송 문제를 오래된 순으로.

        행 잠금은 걸지 않는다. 전송은 몇 분씩 재시도할 수 있어 트랜잭션을
        열어 둔 채 보낼 수 없고, 두 곳에서 동시에 보내는 일은 전송 배치의
        advisory lock이 막는다.

        Parameters:
            purpose (ProblemPurpose): 용도
            limit (int): 최대 개수
            exclude_ids (Collection[uuid.UUID]): 뺄 행. 이번 실행에서 거부된 문제
        """
        stmt = select(GeneratedProblem).where(
            GeneratedProblem.sent_at.is_(None),
            GeneratedProblem.trigger == Trigger.BATCH,
            GeneratedProblem.purpose == purpose,
        )
        if exclude_ids:
            stmt = stmt.where(GeneratedProblem.id.not_in(exclude_ids))
        stmt = stmt.order_by(GeneratedProblem.created_at, GeneratedProblem.id)
        return (await self.session.scalars(stmt.limit(limit))).all()

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
