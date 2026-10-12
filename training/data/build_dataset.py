"""변환 결과를 학습용 jsonl로 만든다.

user에는 실제 서비스가 쓰는 build_prompt 결과를, assistant에는 정답 JSON을 넣는다.
few-shot은 같은 조합의 다른 변환본에서 1개를 가져온다. DB 문제는 쓰지 않는다.

실행: uv run python -m training.data.build_dataset
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from src.core.enums import Category, Difficulty
from src.problem.nodes.generate_problem import build_prompt

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

INPUT_PATH = Path("training/data/converted.jsonl")
TRAIN_PATH = Path("training/data/train.jsonl")
VAL_PATH = Path("training/data/val.jsonl")

FEW_SHOT_COUNT = 1
VAL_RATIO = 0.1
SEED = 42

# few-shot으로 넘길 때 쓰는 필드. render_few_shot이 이 네 개만 읽는다.
FEW_SHOT_FIELDS = (
    "problem_title",
    "problem_content",
    "input_format",
    "output_format",
)


def load_problems() -> list[dict]:
    """변환 결과를 읽는다."""
    lines = INPUT_PATH.read_text(encoding="utf-8").strip().split("\n")
    return [json.loads(line) for line in lines]


def group_by_key(problems: list[dict]) -> dict[tuple, list[dict]]:
    """카테고리와 난이도 조합으로 묶는다."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for problem in problems:
        groups[(problem["category"], problem["difficulty"])].append(problem)
    return groups


def pick_few_shot(problem: dict, groups: dict[tuple, list[dict]]) -> list[dict]:
    """같은 조합의 다른 문제를 참고 문제로 고른다. 없으면 빈 목록."""
    key = (problem["category"], problem["difficulty"])
    others = [p for p in groups[key] if p is not problem]
    if not others:
        return []
    picked = random.sample(others, min(FEW_SHOT_COUNT, len(others)))
    return [{field: p[field] for field in FEW_SHOT_FIELDS} for p in picked]


def to_sample(problem: dict, groups: dict[tuple, list[dict]]) -> dict:
    """문제 하나를 학습 샘플로 만든다."""
    prompt = build_prompt(
        Difficulty(problem["difficulty"]),
        Category(problem["category"]),
        pick_few_shot(problem, groups),
    )
    answer = json.dumps(problem, ensure_ascii=False)
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
    }


def split(samples: list[dict], ratio: float) -> tuple[list[dict], list[dict]]:
    """학습용과 검증용으로 나눈다."""
    shuffled = samples[:]
    random.shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * ratio))
    return shuffled[val_size:], shuffled[:val_size]


def write_jsonl(path: Path, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="학습용 jsonl 생성")
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    args = parser.parse_args()

    random.seed(SEED)

    problems = load_problems()
    groups = group_by_key(problems)
    samples = [to_sample(problem, groups) for problem in problems]
    train, val = split(samples, args.val_ratio)

    write_jsonl(TRAIN_PATH, train)
    write_jsonl(VAL_PATH, val)

    logger.info("학습 %d건 → %s", len(train), TRAIN_PATH)
    logger.info("검증 %d건 → %s", len(val), VAL_PATH)


if __name__ == "__main__":
    main()
