"""외부 데이터셋 문제를 우리 GeneratedProblem 형식으로 변환한다.

난이도와 카테고리는 mapping.py가 정하고, 지문 번역과 나머지 필드 채우기만
Gemini에 맡긴다. 결과는 jsonl로 저장한다. 줄마다 원본 이름(source_name)을 붙여,
다음 변환에서 --skip으로 이미 변환한 문제를 건너뛸 수 있게 한다.

실행: uv run python -m training.data.convert --limit 20
추가: uv run python -m training.data.convert --limit 100 \
        --skip training/data/converted.jsonl --out training/data/converted_more.jsonl
"""

import argparse
import asyncio
import json
import logging
import re
from collections.abc import Iterable
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

# 변환 결과에 붙이는 원본 이름 키. build_dataset이 정답에서 뺀다
SOURCE_NAME_KEY = "source_name"

# 우리 검증은 judge0 출력의 정확한 일치라서, 채점기가 필요한 문제는 학습에 쓰지 않는다
MULTIPLE_ANSWERS = re.compile(
    r"(print|output) any\b"
    r"|any (one )?of them"
    r"|(multiple|several|many) (possible |valid |correct |optimal )?(answers|solutions)"
    r"|any (valid|correct|optimal|suitable) (answer|solution|one)",
    re.IGNORECASE,
)
FLOAT_TOLERANCE = re.compile(
    r"(absolute|relative) (or (absolute|relative) )?error"
    r"|(print|output)[^.\n]{0,60}digits after the decimal point",
    re.IGNORECASE,
)
INTERACTIVE = re.compile(r"\binteractive\b|\bflush\b", re.IGNORECASE)


def needs_checker(description: str) -> bool:
    """정답이 하나로 정해지지 않아 채점기가 필요한 문제인지 확인한다."""
    return any(
        pattern.search(description)
        for pattern in (MULTIPLE_ANSWERS, FLOAT_TOLERANCE, INTERACTIVE)
    )


def is_candidate(row: dict) -> bool:
    """레이팅·태그·예시가 있고, 채점기가 필요 없고, 카테고리로 옮길 수 있는 원본인지."""
    if not (row["cf_rating"] > 0 and row["cf_tags"]):
        return False
    if not row["public_tests"]["input"]:
        return False
    if needs_checker(row["description"]):
        return False
    return to_category(row["cf_tags"]) is not None


def normalize_io(text: str) -> str:
    """줄 끝 공백과 앞뒤 빈 줄을 지운다. 원본 입출력은 끝에 줄바꿈이 붙어 있다."""
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def examples_match_source(problem: GeneratedProblem, row: dict) -> bool:
    """변환한 예시가 모두 원본 예시의 (입력, 출력) 쌍인지 확인한다. 순서는 상관없다."""
    tests = row["public_tests"]
    source = {
        (normalize_io(stdin), normalize_io(expected))
        for stdin, expected in zip(tests["input"], tests["output"], strict=True)
    }
    return all(
        (normalize_io(example.input), normalize_io(example.output)) in source
        for example in problem.problem_examples
    )


def build_prompt(row: dict) -> str | None:
    """원본 한 건으로 변환 프롬프트를 만든다. 매핑이 안 되면 None."""
    category = to_category(row["cf_tags"])
    if category is None:
        return None

    time_limit = row["time_limit"]
    time_limit_s = (
        time_limit["seconds"] + time_limit["nanos"] / 1e9 if time_limit else 0
    ) or DEFAULT_TIME_LIMIT_S
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
            problem = await call_llm_structured(prompt, cfg, GeneratedProblem)
        except Exception as error:
            logger.warning("변환 실패 (%s): %r", row["name"][:40], error)
            return None

    # 예시를 바꾸거나 지어내면 정답 출력을 믿을 수 없으므로 버린다
    if not examples_match_source(problem, row):
        logger.warning("예시가 원본과 다름 (%s)", row["name"][:40])
        return None
    return problem


def select_rows(
    rows: Iterable[dict],
    limit: int,
    offset: int = 0,
    skip_names: frozenset[str] = frozenset(),
) -> list[dict]:
    """
    후보 원본 중에서 변환할 것을 고른다. 조합별로 돌아가며 뽑아 고르게 섞는다.

    Div1/Div2에 같이 나온 문제는 지문이 같으므로 하나만 쓴다.
    skip_names에 있는 문제(이미 변환한 것)는 뽑지 않고,
    그 문제와 지문이 같은 문제도 뽑지 않는다.

    Parameters:
        rows (Iterable[dict]): 원본 행
        limit (int): 뽑을 건수
        offset (int): 앞에서 건너뛸 건수
        skip_names (frozenset[str]): 건너뛸 원본 이름

    Returns:
        list[dict]: 뽑은 원본 행
    """
    candidates = [row for row in rows if is_candidate(row)]
    seen = {
        row["description"].strip() for row in candidates if row["name"] in skip_names
    }

    buckets: dict[tuple, list[dict]] = {}
    for row in candidates:
        if row["name"] in skip_names:
            continue
        description = row["description"].strip()
        if description in seen:
            continue
        seen.add(description)
        key = (to_category(row["cf_tags"]), to_difficulty(row["cf_rating"]))
        buckets.setdefault(key, []).append(row)

    picked: list[dict] = []
    index = 0
    target = limit + offset
    while len(picked) < target:
        added = False
        for rows in buckets.values():
            if index < len(rows):
                picked.append(rows[index])
                added = True
                if len(picked) >= target:
                    break
        if not added:
            break
        index += 1
    return picked[offset:]


def pick_rows(
    limit: int, offset: int = 0, skip_names: frozenset[str] = frozenset()
) -> list[dict]:
    """code_contests를 불러와 변환할 원본을 고른다. 고르는 규칙은 select_rows."""
    ds = load_dataset("deepmind/code_contests", split="train")
    return select_rows(ds, limit, offset, skip_names)


def load_source_names(paths: list[Path]) -> frozenset[str]:
    """
    이전 변환 결과에서 원본 이름을 모은다.

    Raises:
        ValueError: source_name이 없는 줄이 있는 경우 (이름 없이 만든 예전 결과)
    """
    names: set[str] = set()
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            name = json.loads(line).get(SOURCE_NAME_KEY)
            if name is None:
                raise ValueError(f"{path}:{number}에 {SOURCE_NAME_KEY}가 없습니다.")
            names.add(name)
    return frozenset(names)


async def main() -> None:
    parser = argparse.ArgumentParser(description="외부 데이터셋을 우리 형식으로 변환")
    parser.add_argument("--limit", type=int, default=20, help="변환할 건수")
    parser.add_argument("--offset", type=int, default=0, help="건너뛸 건수")
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH, help="출력 경로")
    parser.add_argument(
        "--skip",
        type=Path,
        nargs="*",
        default=[],
        help="이전 변환 결과. 그 안의 원본은 다시 뽑지 않는다",
    )
    args = parser.parse_args()
    # 같은 파일이면 읽기 전에 덮어써 이전 결과를 잃는다
    if args.out.resolve() in {path.resolve() for path in args.skip}:
        parser.error("--out은 --skip에 준 파일과 달라야 합니다.")

    skip_names = load_source_names(args.skip)
    rows = pick_rows(args.limit, args.offset, skip_names)
    logger.info("변환 대상 %d건 (건너뛴 기존 변환 %d건)", len(rows), len(skip_names))

    cfg = LLMConfig(model_name=MODEL_NAME)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    results = await asyncio.gather(*(convert_one(row, cfg, semaphore) for row in rows))

    converted = [
        (row, problem)
        for row, problem in zip(rows, results, strict=True)
        if problem is not None
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row, problem in converted:
            record = {SOURCE_NAME_KEY: row["name"], **problem.model_dump(mode="json")}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("변환 끝: %d/%d건 → %s", len(converted), len(rows), args.out)


if __name__ == "__main__":
    asyncio.run(main())
