import json
from pathlib import Path

import langsmith.utils
import pytest

from evals.common import (
    Row,
    append_row,
    check_meta,
    configure_langsmith,
    load_rows,
    run_concurrently,
    select_pending,
)
from evals.requests import EvalRequest
from src.core.enums import Category, Difficulty


class FakeRow(Row):
    status: str = "ok"


def make_requests(n: int) -> list[EvalRequest]:
    return [
        EvalRequest(
            id=f"v1-{i:04d}", difficulty=Difficulty.LV1, category=Category.DP, seed=i
        )
        for i in range(1, n + 1)
    ]


# ── 결과 파일 ─────────────────────────────────────────────────────────


def test_같은_요청이_여러_줄이면_마지막_줄을_쓴다(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    for row in [
        FakeRow(request_id="a", status="error"),
        FakeRow(request_id="a"),
        FakeRow(request_id="b"),
    ]:
        append_row(path, row)

    rows = load_rows(path, FakeRow)

    assert set(rows) == {"a", "b"}
    assert rows["a"].status == "ok"


def test_결과_파일이_없으면_빈_결과():
    assert load_rows(Path("/nonexistent/rows.jsonl"), FakeRow) == {}


# ── 이어서 돌기 ───────────────────────────────────────────────────────


def test_끝난_요청은_건너뛴다():
    requests = make_requests(3)
    done = {requests[0].id: FakeRow(request_id=requests[0].id, status="error")}

    assert select_pending(requests, done) == requests[1:]


def test_다시_돌릴_결과는_다시_넣는다():
    requests = make_requests(3)
    done = {
        requests[0].id: FakeRow(request_id=requests[0].id, status="error"),
        requests[1].id: FakeRow(request_id=requests[1].id, status="ok"),
    }

    pending = select_pending(requests, done, lambda row: row.status == "error")

    assert pending == [requests[0], requests[2]]


# ── meta ──────────────────────────────────────────────────────────────

META = {
    "model": "Qwen/Qwen3-4B",
    "requests_sha256": "abc",
    "created_at": "2026-10-08T00:00:00+00:00",
}
IDENTITY = ("model", "requests_sha256")


def test_처음_실행이면_meta를_쓴다(tmp_path: Path):
    path = tmp_path / "meta.json"

    check_meta(path, META, IDENTITY)

    assert json.loads(path.read_text()) == META


def test_identity_밖의_값만_다르면_이어서_돌린다(tmp_path: Path):
    path = tmp_path / "meta.json"
    check_meta(path, META, IDENTITY)

    check_meta(path, {**META, "created_at": "2026-10-09T00:00:00+00:00"}, IDENTITY)


def test_identity가_다르면_같은_label로_이어_돌리지_않는다(tmp_path: Path):
    path = tmp_path / "meta.json"
    check_meta(path, META, IDENTITY)

    with pytest.raises(SystemExit, match="model"):
        check_meta(path, {**META, "model": "qwen3-4b-ft"}, IDENTITY)


# ── LangSmith ─────────────────────────────────────────────────────────


def test_트레이스를_평가_프로젝트로_보낸다(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LANGSMITH_PROJECT", "cosmos")
    # 앞선 테스트가 남긴 캐시를 비우고, "cosmos"가 캐시에 들어간 상태에서 시작한다.
    # get_env_var도 캐시라 둘 다 비워야 한다 (.env가 없는 CI에는 'default'가 남는다)
    langsmith.utils.get_env_var.cache_clear()
    langsmith.utils.get_tracer_project.cache_clear()
    assert langsmith.utils.get_tracer_project() == "cosmos"

    try:
        configure_langsmith("cosmos-model-eval")
        assert langsmith.utils.get_tracer_project() == "cosmos-model-eval"
    finally:
        monkeypatch.undo()
        langsmith.utils.get_env_var.cache_clear()
        langsmith.utils.get_tracer_project.cache_clear()


# ── 동시 실행 ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_끝나는_대로_한_줄씩_쌓는다(tmp_path: Path, capsys):
    requests = make_requests(3)
    rows_path = tmp_path / "rows.jsonl"

    async def run_one(request: EvalRequest) -> FakeRow:
        return FakeRow(request_id=request.id)

    rows = await run_concurrently(
        requests,
        run_one,
        rows_path=rows_path,
        concurrency=2,
        describe=lambda row: row.status,
    )

    assert {row.request_id for row in rows} == {r.id for r in requests}
    assert set(load_rows(rows_path, FakeRow)) == {r.id for r in requests}
    assert "[3/3]" in capsys.readouterr().out
