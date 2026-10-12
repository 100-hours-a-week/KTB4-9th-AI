import random

import langsmith.utils
import pytest

from evals import generate as module
from evals.common import GEMINI
from evals.generate import (
    GenerationRow,
    generate_one,
    is_model_error,
    is_retryable,
    parse_args,
)
from evals.requests import EvalRequest
from src.client.llm import LLMConfig, LocalModel, current_local_model
from src.core.enums import Category, Difficulty
from src.core.exception import LLMOutputParseError
from src.problem.state import GraphState, ProblemExample

REQUEST = EvalRequest(
    id="v1-0001", difficulty=Difficulty.LV2, category=Category.DP, seed=7
)
LOCAL = LocalModel(model_name="qwen3-4b-ft")


@pytest.fixture(autouse=True)
def no_tracing(monkeypatch: pytest.MonkeyPatch):
    """테스트가 LangSmith로 트레이스를 보내지 않게 한다."""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    langsmith.utils.get_env_var.cache_clear()
    yield
    monkeypatch.undo()
    langsmith.utils.get_env_var.cache_clear()


def fix_generate_problem(monkeypatch: pytest.MonkeyPatch, node) -> None:
    monkeypatch.setattr(module, "generate_problem", node)


async def fake_problem(state: GraphState) -> dict:
    return {
        "difficulty": state.requested_difficulty,
        "category": state.requested_category,
        "problem_title": "합이 M인 쌍의 개수",
        "problem_examples": [ProblemExample(input="5 6\n1 2 3 4 5", output="2")],
        "node_models": {"generate_problem": LLMConfig(model_name="gemini-3.8-flash")},
    }


def error_row(error: str) -> GenerationRow:
    return GenerationRow.model_validate(
        {
            "request_id": "a",
            "label": "t",
            "model": GEMINI,
            "difficulty": "LV1",
            "category": "DP",
            "status": "error",
            "error": error,
            "seconds": 1.0,
            "started_at": "2026-10-08T00:00:00Z",
        }
    )


@pytest.mark.asyncio
async def test_생성_결과를_그래프에_다시_넣을_수_있는_JSON으로_남긴다(monkeypatch):
    fix_generate_problem(monkeypatch, fake_problem)

    row = await generate_one(REQUEST, None, label="test")

    assert row.status == "ok", row.error
    assert row.model == GEMINI
    assert row.output["problem_title"] == "합이 M인 쌍의 개수"
    assert row.output["problem_examples"] == [
        {"input": "5 6\n1 2 3 4 5", "output": "2", "description": None}
    ]
    assert "node_models" not in row.output, "Gemini 이름이 남으면 헷갈린다"
    GraphState(
        **{**row.output, "requested_difficulty": "LV2", "requested_category": "DP"}
    )


@pytest.mark.asyncio
async def test_로컬_모델로_생성한다(monkeypatch):
    seen = []

    async def node(state: GraphState) -> dict:
        seen.append(current_local_model())
        return await fake_problem(state)

    fix_generate_problem(monkeypatch, node)

    row = await generate_one(REQUEST, LOCAL, label="test")

    assert seen == [LOCAL]
    assert row.model == "qwen3-4b-ft"
    assert current_local_model() is None, "생성이 끝나면 스위치가 꺼져야 한다"


@pytest.mark.asyncio
async def test_요청마다_같은_seed로_시작한다(monkeypatch):
    draws = []

    async def node(state: GraphState) -> dict:
        draws.append(random.random())
        return await fake_problem(state)

    fix_generate_problem(monkeypatch, node)

    await generate_one(REQUEST, None, label="a")
    await generate_one(REQUEST, None, label="b")

    assert draws[0] == draws[1]


@pytest.mark.asyncio
async def test_형식_실패는_모델_에러로_남긴다(monkeypatch):
    async def node(state: GraphState) -> dict:
        raise LLMOutputParseError("구조화 응답 해석 실패 (finish_reason=length)")

    fix_generate_problem(monkeypatch, node)

    row = await generate_one(REQUEST, LOCAL, label="test")

    assert row.status == "error"
    assert row.output is None
    assert is_model_error(row)
    assert not is_retryable(row), "모델 탓인 실패는 결과이므로 다시 돌지 않는다"


@pytest.mark.asyncio
async def test_연결_실패는_인프라_에러로_남긴다(monkeypatch):
    async def node(state: GraphState) -> dict:
        raise ConnectionError("ngrok offline")

    fix_generate_problem(monkeypatch, node)

    row = await generate_one(REQUEST, LOCAL, label="test")

    assert row.status == "error"
    assert row.error.startswith("ConnectionError")
    assert not is_model_error(row)
    assert is_retryable(row)


def test_성공한_결과는_다시_돌지_않는다():
    row = error_row("x").model_copy(update={"status": "ok", "error": None})

    assert not is_retryable(row)


def test_모델을_비우면_Gemini로_생성한다():
    args = parse_args(["--label", "gemini-baseline"])

    assert args.model is None


def test_chat_template_kwargs는_JSON으로_읽는다():
    args = parse_args(
        [
            "--label",
            "q",
            "--model",
            "Qwen/Qwen3-4B",
            "--chat-template-kwargs",
            '{"enable_thinking": false}',
        ]
    )

    assert args.chat_template_kwargs == {"enable_thinking": False}


@pytest.mark.parametrize(
    "argv",
    [
        ["--label", "a/b"],  # 폴더 경로가 되면 안 된다
        ["--label", "x", "--temperature", "0.7"],  # Gemini에는 적용되지 않는다
        ["--label", "x", "--chat-template-kwargs", "{}"],
    ],
)
def test_잘못된_인자는_거절한다(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)
