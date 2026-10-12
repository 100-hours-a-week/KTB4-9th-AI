from pathlib import Path

from evals.requests import EvalRequest, load_requests
from src.core.enums import Category, Difficulty


def test_load_requests_round_trips(tmp_path: Path) -> None:
    requests = [
        EvalRequest(
            id="v1-0001", difficulty=Difficulty.LV1, category=Category.DP, seed=1
        ),
        EvalRequest(
            id="v1-0002", difficulty=Difficulty.LV3, category=Category.TREE, seed=2
        ),
    ]
    path = tmp_path / "requests.jsonl"
    path.write_text(
        "".join(request.model_dump_json() + "\n" for request in requests) + "\n",
        encoding="utf-8",
    )

    assert load_requests(path) == requests
