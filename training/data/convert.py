"""외부 데이터셋 문제를 우리 GeneratedProblem 형식으로 변환한다.

난이도와 카테고리는 mapping.py가 정하고, 지문 번역과 나머지 필드 채우기만
Gemini에 맡긴다. 결과는 jsonl로 저장한다.

실행: uv run python -m training.data.convert --limit 20
"""

import argparse
import asyncio
import logging
from pathlib import Path

from datasets import load_dataset

from src.client.llm import LLMConfig, call_llm_structured
from src.core.enums import Language
from src.problem.nodes.generate_problem import GeneratedProblem
from src.problem.state import (
    MAX_CATEGORY_REASON_LENGTH,
    MAX_CONTENT_LENGTH,
    MAX_TITLE_LENGTH,
)
from training.data.convert_prompt import CONVERT_PROMPT, render_examples
from training.data.mapping import to_category, to_difficulty

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-3.8-flash"
OUTPUT_PATH = Path("training/data/converted.jsonl")

DEFAULT_TIME_LIMIT_S = 2
DEFAULT_MEMORY_LIMIT_MB = 256
CONCURRENCY = 5


def build_prompt(row: dict) -> str | None:
    """원본 한 건으로 변환 프롬프트를 만든다. 매핑이 안 되면 None."""
    category = to_category(row["cf_tags"])
    if category is None:
        return None

    time_limit = row["time_limit"]
    time_limit_s = time_limit["seconds"] if time_limit else DEFAULT_TIME_LIMIT_S
    memory_bytes = row["memory_limit_bytes"]
    memory_limit_mb = (
        memory_bytes // 1_000_000 if memory_bytes else DEFAULT_MEMORY_LIMIT_MB
    )

    return CONVERT_PROMPT.format(
        name=row["name"],
        description=row["description"],
        examples=render_examples(row["public_tests"]),
        time_limit_s=time_limit_s,
        memory_limit_mb=memory_limit_mb,
        difficulty=to_difficulty(row["cf_rating"]).value,
        category=category.value,
        languages=", ".join(lang.value for lang in Language),
        max_title=MAX_TITLE_LENGTH,
        max_content=MAX_CONTENT_LENGTH,
        max_reason=MAX_CATEGORY_REASON_LENGTH,
    )


async def convert_one(
    row: dict, cfg: LLMConfig, semaphore: asyncio.Semaphore
) -> GeneratedProblem | None:
    """한 건을 변환한다. 실패하면 None."""
    prompt = build_prompt(row)
    if prompt is None:
        return None

    async with semaphore:
        try:
            return await call_llm_structured(prompt, cfg, GeneratedProblem)
        except Exception as error:
            logger.warning("변환 실패 (%s): %r", row["name"][:40], error)
            return None


def pick_rows(limit: int) -> list[dict]:
    """변환할 원본을 고른다. 난이도와 카테고리가 고르게 섞이도록 돌아가며 뽑는다."""
    ds = load_dataset("deepmind/code_contests", split="train")

    buckets: dict[tuple, list[dict]] = {}
    for row in ds:
        if not (row["cf_rating"] > 0 and row["cf_tags"]):
            continue
        if not row["public_tests"]["input"]:
            continue
        category = to_category(row["cf_tags"])
        if category is None:
            continue
        key = (category, to_difficulty(row["cf_rating"]))
        buckets.setdefault(key, []).append(row)

    picked: list[dict] = []
    index = 0
    while len(picked) < limit:
        added = False
        for rows in buckets.values():
            if index < len(rows):
                picked.append(rows[index])
                added = True
                if len(picked) >= limit:
                    break
        if not added:
            break
        index += 1
    return picked


async def main() -> None:
    parser = argparse.ArgumentParser(description="외부 데이터셋을 우리 형식으로 변환")
    parser.add_argument("--limit", type=int, default=20, help="변환할 건수")
    args = parser.parse_args()

    rows = pick_rows(args.limit)
    logger.info("변환 대상 %d건", len(rows))

    cfg = LLMConfig(model_name=MODEL_NAME)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    results = await asyncio.gather(*(convert_one(row, cfg, semaphore) for row in rows))

    problems = [p for p in results if p is not None]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for problem in problems:
            f.write(problem.model_dump_json() + "\n")

    logger.info("변환 끝: %d/%d건 → %s", len(problems), len(rows), OUTPUT_PATH)


if __name__ == "__main__":
    asyncio.run(main())
