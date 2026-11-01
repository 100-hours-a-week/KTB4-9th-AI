import json

import pytest

from src.problem.nodes.generate_problem import GeneratedProblem
from src.problem.state import ProblemExample
from training.data import convert


def make_row(name: str, *, description: str | None = None, rating: int = 1000) -> dict:
    """code_contests 행을 흉내 낸다. 기본값은 변환 후보가 되는 행이다."""
    return {
        "name": name,
        "description": description or f"{name} 지문",
        "cf_rating": rating,
        "cf_tags": ["greedy"],
        "public_tests": {"input": ["3\n1 2 3\n"], "output": ["6\n"]},
    }


def make_problem(*pairs: tuple[str, str]) -> GeneratedProblem:
    """예시만 채운 변환 결과."""
    examples = [
        ProblemExample(input=stdin, description="계산 과정", output=expected)
        for stdin, expected in pairs
    ]
    return GeneratedProblem.model_construct(problem_examples=examples)


@pytest.mark.parametrize(
    "description",
    [
        "If there are several possible solutions, print any of them.",
        "The answer is accepted if absolute or relative error is at most 1e-6.",
        "Print the answer with at least 5 digits after the decimal point.",
        "This is an interactive problem. Do not forget to flush the output.",
    ],
)
def test_채점기가_필요한_지문을_골라낸다(description):
    assert convert.needs_checker(description)


@pytest.mark.parametrize(
    "description",
    [
        "Print the minimum number of operations.",
        "The probabilities are given with 2 digits after the decimal point.",
    ],
)
def test_정답이_하나인_지문은_남긴다(description):
    assert not convert.needs_checker(description)


def test_원본_예시와_같으면_통과한다():
    row = make_row("A")
    row["public_tests"] = {"input": ["1\n", "2\n"], "output": ["1\n", "4\n"]}

    # 순서가 바뀌고 끝 줄바꿈이 없어도 같은 쌍이다
    assert convert.examples_match_source(make_problem(("2", "4"), ("1", "1")), row)


def test_원본에_없는_예시가_있으면_실패한다():
    row = make_row("A")

    assert not convert.examples_match_source(make_problem(("3\n1 2 3", "7")), row)


def test_같은_지문은_한_번만_뽑는다():
    rows = [
        make_row("115_A. Party", description="같은 지문"),
        make_row("116_C. Party", description="같은 지문"),
        make_row("1_A. Other"),
    ]

    picked = convert.select_rows(rows, limit=10)

    assert [row["name"] for row in picked] == ["115_A. Party", "1_A. Other"]


def test_건너뛸_문제와_지문이_같은_문제도_뽑지_않는다():
    rows = [
        make_row("115_A. Party", description="같은 지문"),
        make_row("116_C. Party", description="같은 지문"),
        make_row("1_A. Other"),
    ]

    picked = convert.select_rows(rows, limit=10, skip_names=frozenset({"116_C. Party"}))

    assert [row["name"] for row in picked] == ["1_A. Other"]


def test_후보가_아닌_원본은_뽑지_않는다():
    rows = [
        make_row("no rating", rating=0),
        make_row("any", description="If there are multiple answers, print any."),
        make_row("ok"),
    ]

    picked = convert.select_rows(rows, limit=10)

    assert [row["name"] for row in picked] == ["ok"]


def test_이전_변환_결과에서_원본_이름을_모은다(tmp_path):
    path = tmp_path / "converted.jsonl"
    lines = [{"source_name": "A", "problem_title": "가"}, {"source_name": "B"}]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    assert convert.load_source_names([path]) == frozenset({"A", "B"})


def test_원본_이름이_없는_변환_결과는_거절한다(tmp_path):
    path = tmp_path / "old.jsonl"
    path.write_text(json.dumps({"problem_title": "가"}) + "\n")

    with pytest.raises(ValueError, match="source_name"):
        convert.load_source_names([path])
