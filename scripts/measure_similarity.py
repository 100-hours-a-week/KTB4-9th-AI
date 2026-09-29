"""중복 판정 임계값을 정하기 위한 측정 스크립트.

두 곳을 본다.
1. problem_embeddings — 저장된 문제들끼리의 유사도 분포
2. discarded_problems — 중복으로 걸러진 문제들의 유사도

2번이 중요하다. 색인에는 임계값을 통과한 문제만 남아
최댓값이 기준선 바로 아래에서 잘리기 때문에, 1번만 보면
"기준이 적절하다"고 오독하게 된다.

실행: uv run python -m scripts/measure_similarity.py
"""

import asyncio
import re
import statistics

from sqlalchemy import text

from src.db.session import session_scope

SIMILARITY_PATTERN = re.compile(r"유사도 ([\d.]+)")

# 같은 카테고리끼리의 모든 쌍. 운영 코드의 find_similar와 같은 비교 범위다.
SAME_CATEGORY_SQL = """
SELECT 1 - (a.embedding <=> b.embedding) AS sim
FROM problem_embeddings a
JOIN problem_embeddings b
  ON a.category = b.category
 AND a.id < b.id
 AND a.embedding_model = b.embedding_model
"""

# 음성 기준선. 서로 다른 카테고리는 중복일 수 없다.
DIFF_CATEGORY_SQL = """
SELECT 1 - (a.embedding <=> b.embedding) AS sim
FROM problem_embeddings a
JOIN problem_embeddings b
  ON a.category <> b.category
 AND a.id < b.id
 AND a.embedding_model = b.embedding_model
"""

DISCARDED_SQL = """
SELECT discard_detail
FROM discarded_problems
WHERE discard_reason = 'DUPLICATE'
"""


def describe(name: str, values: list[float]) -> None:
    """유사도 목록의 요약 통계를 출력한다."""
    if not values:
        print(f"{name}: 표본 없음")
        return

    ordered = sorted(values)
    p95 = ordered[int(len(ordered) * 0.95) - 1] if len(ordered) >= 20 else ordered[-1]
    print(
        f"{name}: {len(values)}개 "
        f"최소 {min(values):.3f} / 중앙 {statistics.median(values):.3f} / "
        f"p95 {p95:.3f} / 최대 {max(values):.3f}"
    )


def histogram(name: str, values: list[float], width: float = 0.05) -> None:
    """0.05 폭으로 구간을 나눠 분포를 출력한다."""
    if not values:
        return

    print(f"\n{name} 분포")
    buckets: dict[float, int] = {}
    for value in values:
        key = round(value // width * width, 2)
        buckets[key] = buckets.get(key, 0) + 1

    for key in sorted(buckets):
        count = buckets[key]
        ratio = count / len(values)
        bar = "#" * int(ratio * 50)
        print(f"  {key:.2f}~{key + width:.2f}  {count:4d} ({ratio:5.1%}) {bar}")


async def fetch_similarities(sql: str) -> list[float]:
    """쿼리 결과를 유사도 목록으로 가져온다."""
    async with session_scope() as session:
        rows = await session.execute(text(sql))
        return [row[0] for row in rows]


async def fetch_discarded() -> list[float]:
    """폐기 로그에서 유사도를 파싱한다.

    check_duplicate가 사유 문자열에 유사도를 숫자로 기록하므로,
    다시 계산하지 않고 생성 당시 게이트가 본 값을 그대로 쓴다.
    """
    async with session_scope() as session:
        rows = await session.execute(text(DISCARDED_SQL))
        values: list[float] = []
        for (detail,) in rows:
            match = SIMILARITY_PATTERN.search(detail or "")
            if match:
                values.append(float(match.group(1)))
        return values


async def main() -> None:
    same = await fetch_similarities(SAME_CATEGORY_SQL)
    diff = await fetch_similarities(DIFF_CATEGORY_SQL)
    discarded = await fetch_discarded()

    print("=" * 60)
    print("저장된 문제 (임계값을 통과한 것만 남아 있음)")
    print("=" * 60)
    describe("같은 카테고리", same)
    describe("다른 카테고리", diff)

    print()
    print("=" * 60)
    print("폐기된 문제 (검열되지 않은 표본)")
    print("=" * 60)
    describe("중복 폐기", discarded)

    histogram("같은 카테고리", same)
    histogram("중복 폐기", discarded)


if __name__ == "__main__":
    asyncio.run(main())
