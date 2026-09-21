import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Enum, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from src.enum import Category, Difficulty, DiscardReason, SeedSource

EMBEDDING_DIM = 768

# JSONB 컬럼 규칙: Pydantic 스키마의 model_dump(mode="json") 결과(snake_case)그대로 저장
JsonList = list[dict[str, Any]]


class Base(DeclarativeBase):
    pass


def _enum_col(enum_cls: type[StrEnum]) -> Enum:
    return Enum(
        enum_cls,
        native_enum=False,
        length=32,
        values_callable=lambda e: [m.value for m in e],
    )


# ── 1. 생성 버퍼 ───────────────────────────────────────────────────────


class GeneratedProblem(Base):
    """Spring 전송 대기 버퍼. 전송 후에도 문제 전문을 보관한다.

    컬럼 구성은 스키마의 Problem과 1:1로 맞춘다.
    """

    __tablename__ = "generated_problems"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)

    requested_difficulty: Mapped[Difficulty] = mapped_column(_enum_col(Difficulty))

    # 분류
    category: Mapped[Category] = mapped_column(_enum_col(Category), index=True)
    difficulty: Mapped[Difficulty] = mapped_column(_enum_col(Difficulty))
    category_select_reason: Mapped[str] = mapped_column(Text)

    # 지문
    problem_title: Mapped[str] = mapped_column(String(255))
    problem_description: Mapped[str] = mapped_column(Text)
    input_format: Mapped[str] = mapped_column(Text)
    output_format: Mapped[str] = mapped_column(Text)

    # 구조화 데이터
    input_constraints: Mapped[JsonList] = mapped_column(JSONB)  # list[InputConstraint]
    execution_limits: Mapped[JsonList] = mapped_column(JSONB)  # list[ExecutionLimit]
    problem_examples: Mapped[JsonList] = mapped_column(JSONB)  # list[ProblemExample]
    hidden_test_cases: Mapped[JsonList] = mapped_column(
        JSONB
    )  # list[HiddenTestCase], 20개
    solution_codes: Mapped[JsonList] = mapped_column(JSONB)  # list[SolutionCode]
    hint_comments: Mapped[JsonList] = mapped_column(JSONB)  # list[HintComment]
    solution_keywords: Mapped[list[str]] = mapped_column(JSONB)  # 채점 키워드

    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        # 미전송 재고 조회용 부분 인덱스
        Index(
            "ix_generated_problems_pending",
            "category",
            "difficulty",
            postgresql_where=sent_at.is_(None),
        ),
    )


# ── 2. 폐기 로그 ───────────────────────────────────────────────────────


class DiscardedProblem(Base):
    """검증에서 떨어진 결과. 사유별 분석용."""

    __tablename__ = "discarded_problems"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)

    requested_category: Mapped[Category] = mapped_column(
        _enum_col(Category), index=True
    )
    requested_difficulty: Mapped[Difficulty] = mapped_column(_enum_col(Difficulty))

    stage: Mapped[str] = mapped_column(String(64))  # 어느 노드에서 걸렸는지
    discard_reason: Mapped[DiscardReason] = mapped_column(
        _enum_col(DiscardReason), index=True
    )
    discard_detail: Mapped[str | None] = mapped_column(Text)  # 사람이 읽을 사유
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)  # 원본 보관

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ── 3. 임베딩 ──────────────────────────────────────────────────────────


class ProblemEmbedding(Base):
    """중복 검사용. 지문 전체가 아니라 알고리즘 핵심만 임베딩한다."""

    __tablename__ = "problem_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    problem_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("generated_problems.id", ondelete="CASCADE"), unique=True
    )

    category: Mapped[Category] = mapped_column(
        _enum_col(Category), index=True
    )  # 비교 범위 한정
    algorithm_core: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    embedding_model: Mapped[str] = mapped_column(String(128))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ── 4. few-shot 시드 ───────────────────────────────────────────────────


class FewshotSeed(Base):
    """프롬프트에 끼워 넣을 예시 문제.

    직접 작성, 외부 문제, 생성 결과 승격 세 경로로 들어온다.
    승격은 복사이므로 원본 버퍼 행이 지워져도 시드는 남는다.
    생성 단계 프롬프트용이라 지문·제약·예시까지만 필수, 코드는 선택.
    """

    __tablename__ = "fewshot_seeds"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)

    # 출처
    source: Mapped[SeedSource] = mapped_column(_enum_col(SeedSource))

    # 분류
    category: Mapped[Category] = mapped_column(_enum_col(Category))
    difficulty: Mapped[Difficulty] = mapped_column(_enum_col(Difficulty))

    # 지문
    problem_title: Mapped[str] = mapped_column(String(255))
    problem_description: Mapped[str] = mapped_column(Text)
    input_format: Mapped[str] = mapped_column(Text)
    output_format: Mapped[str] = mapped_column(Text)

    # 구조화 데이터
    input_constraints: Mapped[JsonList] = mapped_column(JSONB)
    execution_limits: Mapped[JsonList] = mapped_column(JSONB)
    problem_examples: Mapped[JsonList] = mapped_column(JSONB)
    solution_codes: Mapped[JsonList | None] = mapped_column(JSONB)

    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("ix_fewshot_lookup", "category", "difficulty", "is_active"),
    )
