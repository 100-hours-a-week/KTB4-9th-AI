"""fewshot_seeds 테이블에 기본 80개 시드 데이터를 적재하는 모듈."""

import asyncio
import json
import logging
import uuid
from pathlib import Path

from sqlalchemy import delete

from src.core.enums import Category, Difficulty, SeedSource
from src.db.models import FewshotSeed
from src.db.session import close_engine, session_scope

logger = logging.getLogger(__name__)


async def seed_fewshot(jsonl_path: Path | None = None) -> int:
    """JSONL 파일에서 Few-shot 시드를 읽어 DB에 적재한다."""
    if jsonl_path is None:
        jsonl_path = Path(__file__).resolve().parents[2] / "sql" / "fewshot_seeds.jsonl"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"시드 파일이 존재하지 않습니다: {jsonl_path}")

    with jsonl_path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    async with session_scope() as session:
        await session.execute(delete(FewshotSeed))

        for row in rows:
            raw_id = row.get("id")
            parsed_id = uuid.UUID(raw_id) if isinstance(raw_id, str) else raw_id
            seed = FewshotSeed(
                id=parsed_id,
                source=SeedSource(row["source"]),
                source_ref=row.get("source_ref"),
                category=Category(row["category"]),
                difficulty=Difficulty(row["difficulty"]),
                problem_title=row["problem_title"],
                problem_content=row["problem_content"],
                input_format=row["input_format"],
                output_format=row["output_format"],
                input_constraints=row.get("input_constraints") or [],
                execution_limits=row.get("execution_limits") or [],
                problem_examples=row.get("problem_examples") or [],
                solution_codes=row.get("solution_codes"),
                is_active=row.get("is_active", True),
            )
            session.add(seed)

    logger.info("성공적으로 %d개의 few-shot 시드를 적재했습니다.", len(rows))
    return len(rows)


def main() -> None:
    """CLI 진입점."""
    logging.basicConfig(level=logging.INFO)
    try:
        count = asyncio.run(seed_fewshot())
        print(f"Few-shot 시드 적재 완료: {count}건")
    finally:
        asyncio.run(close_engine())


if __name__ == "__main__":
    main()
