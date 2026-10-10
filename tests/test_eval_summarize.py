import json
from pathlib import Path

import pytest

from evals.common import append_row
from evals.generate import GenerationRow
from evals.requests import EvalRequest
from evals.score import ScoreRow, generation_sha
from evals.summarize import (
    FORMAT_FAILED,
    INFRA,
    NOT_GENERATED,
    NOT_SCORED,
    PASSED,
    LabelResult,
    Outcome,
    classify,
    collect,
    count_duplicates,
    limit_problems,
    overview_table,
    paired_table,
    render,
    wilson_interval,
)
from src.core.enums import Category, Difficulty

LIMITS = [{"language": "PYTHON", "time_limit_ms": 2000, "memory_limit_kb": 262144}]
OUTPUT = {
    "problem_title": "합이 M인 쌍의 개수",
    "algorithm_core": "해시맵으로 쌍을 센다",
    "execution_limits": LIMITS,
}


def request(n: int, category: Category = Category.DP) -> EvalRequest:
    return EvalRequest(
        id=f"v1-{n:04d}", difficulty=Difficulty.LV1, category=category, seed=n
    )


def generation(
    req: EvalRequest, *, error: str | None = None, output: dict | None = None
) -> GenerationRow:
    return GenerationRow.model_validate(
        {
            "request_id": req.id,
            "label": "t",
            "model": "m",
            "difficulty": req.difficulty,
            "category": req.category,
            "status": "error" if error else "ok",
            "error": error,
            "seconds": 10.0,
            "output": None if error else (output or OUTPUT),
            "started_at": "2026-10-08T00:00:00Z",
        }
    )


def score(
    req: EvalRequest,
    status: str = "passed",
    stage: str | None = None,
    reason: str | None = None,
    output: dict | None = None,
    attempts: int = 1,
) -> ScoreRow:
    return ScoreRow.model_validate(
        {
            "request_id": req.id,
            "label": "t",
            "model": "m",
            "difficulty": req.difficulty,
            "category": req.category,
            "generation_sha": generation_sha(output or OUTPUT),
            "status": status,
            "discard_stage": stage,
            "discard_reason": reason,
            "seconds": 5.0,
            "node_seconds": {},
            "node_calls": {"generate_ref_code": attempts},
            "started_at": "2026-10-08T00:00:00Z",
        }
    )


# ── 요청 하나의 결과 ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("gen", "scr", "expected"),
    [
        (None, None, NOT_GENERATED),
        ("format", None, FORMAT_FAILED),
        ("infra", None, INFRA),
        ("ok", None, NOT_SCORED),
        ("ok", "stale", NOT_SCORED),
        ("ok", "llm_error", INFRA),
        ("ok", "passed", PASSED),
        ("ok", "mismatch", "semantic_validate:EXAMPLE_MISMATCH"),
    ],
)
def test_요청_결과를_분류한다(gen, scr, expected):
    req = request(1)
    generations = {
        None: None,
        "format": generation(req, error="LLMOutputParseError: 형식"),
        "infra": generation(req, error="APIConnectionError: 끊김"),
        "ok": generation(req),
    }
    scores = {
        None: None,
        "stale": score(req, output={"problem_title": "예전 생성"}),
        "llm_error": score(req, "discarded", "semantic_validate", "LLM_ERROR"),
        "passed": score(req),
        "mismatch": score(req, "discarded", "semantic_validate", "EXAMPLE_MISMATCH"),
    }

    assert classify(req, generations[gen], scores[scr]) == expected


# ── 통계 ──────────────────────────────────────────────────────────────


def test_통과율_구간은_0과_1_사이에_있다():
    assert wilson_interval(0, 0) == (0.0, 0.0)
    low, high = wilson_interval(5, 10)
    assert 0.2 < low < 0.3 and 0.7 < high < 0.8
    assert wilson_interval(10, 10)[1] == 1.0


def label_result(label: str, results: list[str]) -> LabelResult:
    outcomes = []
    for n, result in enumerate(results, start=1):
        req = request(n)
        outcomes.append(
            Outcome(
                request=req,
                result=result,
                generation=generation(req),
                score=score(req),
            )
        )
    return LabelResult(label=label, model="m", outcomes=outcomes)


def test_미확정_요청은_분모에서_뺀다():
    result = label_result(
        "a", [PASSED, PASSED, "semantic_validate:EXAMPLE_MISMATCH", INFRA, NOT_SCORED]
    )

    table = overview_table([result])

    assert "| 3/5 |" in table
    assert "**67%**" in table


def test_실행_제한이_비정상이면_실질_통과에서_뺀다():
    result = label_result("a", [PASSED, PASSED])
    tiny = OUTPUT | {
        "execution_limits": [
            {"language": "PYTHON", "time_limit_ms": 1000, "memory_limit_kb": 1024}
        ]
    }
    result.outcomes[1].generation = generation(request(2), output=tiny)

    table = overview_table([result])

    assert result.outcomes[1].limit_problems == ["PYTHON 메모리 1024KB"]
    assert not result.outcomes[1].usable
    assert "| 100% | 1 (50%) | **50%**" in table


@pytest.mark.parametrize(
    ("memory", "time_ms", "expected"),
    [
        (262144, 2000, []),
        (32768, 100, []),
        (1024, 2000, ["PYTHON 메모리 1024KB"]),
        (4_194_304, 2000, ["PYTHON 메모리 4194304KB"]),
        (262144, 10, ["PYTHON 시간 10ms"]),
    ],
)
def test_실행_제한_범위를_검사한다(memory, time_ms, expected):
    output = {
        "execution_limits": [
            {"language": "PYTHON", "time_limit_ms": time_ms, "memory_limit_kb": memory}
        ]
    }

    assert limit_problems(output) == expected


def test_같은_요청끼리_비교한다():
    base = label_result("gemini", [PASSED, PASSED, FORMAT_FAILED, INFRA])
    other = label_result("qwen", [PASSED, FORMAT_FAILED, PASSED, PASSED])

    table = paired_table([base, other])

    # 4번 요청은 기준이 미확정이라 짝에서 빠진다
    assert "| qwen | 3 | 1 | 1 | 1 | 0 |" in table


# ── 다양성 ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_같은_카테고리의_비슷한_문제만_중복으로_센다(tmp_path: Path):
    vectors = {"a": [1.0, 0.0], "a2": [0.95, 0.312], "b": [0.0, 1.0]}
    calls = []

    async def embed(text: str) -> list[float]:
        calls.append(text)
        return vectors[text]

    outcomes = []
    for n, (core, category) in enumerate(
        [
            ("a", Category.DP),
            ("a2", Category.DP),
            ("a2", Category.TREE),
            ("b", Category.DP),
        ],
        start=1,
    ):
        req = request(n, category)
        outcomes.append(
            Outcome(req, PASSED, generation(req, output={"algorithm_core": core}))
        )
    result = LabelResult(label="x", model="m", outcomes=outcomes)
    cache = tmp_path / "embeddings.json"

    await count_duplicates(result, embed, cache)

    assert (result.duplicates, result.duplicate_pool) == (1, 4), "DP의 a2만 중복"
    assert len(calls) == 3, "같은 생성 결과는 한 번만 임베딩한다"

    calls.clear()
    await count_duplicates(result, embed, cache)
    assert calls == [], "캐시가 있으면 다시 부르지 않는다"


# ── 파일에서 읽기 ─────────────────────────────────────────────────────


def test_결과_폴더에서_모아_표를_만든다(tmp_path: Path):
    requests_path = tmp_path / "requests.jsonl"
    reqs = [request(1), request(2), request(3)]
    requests_path.write_text("".join(r.model_dump_json() + "\n" for r in reqs))
    out_dir = tmp_path / "qwen"
    out_dir.mkdir()
    (out_dir / "generate_meta.json").write_text(
        json.dumps({"model": "Qwen/Qwen3-4B", "requests": str(requests_path)})
    )
    append_row(out_dir / "generations.jsonl", generation(reqs[0]))
    append_row(
        out_dir / "generations.jsonl",
        generation(reqs[1], error="LLMOutputParseError: x"),
    )
    append_row(out_dir / "scores.jsonl", score(reqs[0]))

    result = collect("qwen", tmp_path)
    text = render([result])

    assert [o.result for o in result.outcomes] == [PASSED, FORMAT_FAILED, NOT_GENERATED]
    assert "| qwen | Qwen/Qwen3-4B | 2/3 |" in text
    assert "## 폐기 사유" in text
