import json

import pytest

from src.core.enums import Category, Difficulty
from training.data import build_dataset


def make_problems(count: int) -> list[dict]:
    """같은 조합의 가짜 문제를 만든다."""
    return [
        {
            "category": Category.GREEDY.value,
            "difficulty": Difficulty.LV1.value,
            "problem_title": f"문제 {idx:02d}",
            "problem_content": f"지문 {idx:02d}",
            "input_format": f"입력 {idx:02d}",
            "output_format": f"출력 {idx:02d}",
        }
        for idx in range(count)
    ]


def content_of(sample: dict) -> str:
    """샘플의 정답에서 지문을 꺼낸다."""
    return json.loads(sample["messages"][1]["content"])["problem_content"]


@pytest.fixture
def run_main(monkeypatch, tmp_path):
    """가짜 문제로 main을 돌리고 학습용·검증용 샘플을 돌려준다."""

    def run(problems: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
        train_path = tmp_path / f"train_{seed}.jsonl"
        val_path = tmp_path / f"val_{seed}.jsonl"

        monkeypatch.setattr(build_dataset, "load_problems", lambda: problems)
        monkeypatch.setattr(build_dataset, "TRAIN_PATH", train_path)
        monkeypatch.setattr(build_dataset, "VAL_PATH", val_path)
        monkeypatch.setattr(build_dataset, "SEED", seed)
        monkeypatch.setattr("sys.argv", ["build_dataset"])

        build_dataset.main()

        def read(path):
            lines = path.read_text(encoding="utf-8").strip().split("\n")
            return [json.loads(line) for line in lines]

        return read(train_path), read(val_path)

    return run


def test_검증용_문제가_어떤_프롬프트에도_참고_문제로_들어가지_않는다(run_main):
    for seed in range(20):
        train, val = run_main(make_problems(30), seed)

        prompts = [sample["messages"][0]["content"] for sample in train + val]
        val_contents = [content_of(sample) for sample in val]

        for content in val_contents:
            for prompt in prompts:
                assert content not in prompt, f"seed={seed}"


def test_참고_문제로_자기_자신이_들어가지_않는다(run_main):
    for seed in range(20):
        train, _ = run_main(make_problems(30), seed)

        for sample in train:
            prompt = sample["messages"][0]["content"]
            assert content_of(sample) not in prompt, f"seed={seed}"
