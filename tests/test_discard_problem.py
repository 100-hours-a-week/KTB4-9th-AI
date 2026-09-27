import contextlib

import pytest

from src.core.enums import Category, DiscardReason
from src.problem.nodes import discard_problem as module
from src.problem.nodes.discard_problem import build_discard_record, discard_problem
from src.problem.state import GraphState
from tests import fake_nodes


async def make_state(**updates) -> GraphState:
    """정적 검증에서 폐기된 상태."""
    base = GraphState(requested_difficulty="LV2", requested_category="HASH_TABLE")
    state = base.model_copy(update=await fake_nodes.generate_problem(base))
    return state.model_copy(
        update={
            "is_discarded": True,
            "discard_reason": DiscardReason.MISMATCH_LABEL,
            "discard_detail": "요청 LV2 / 생성 LV5",
            "discard_stage": "static_validate",
            **updates,
        }
    )


def fix_repository(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """DB를 건드리지 않고 넘어온 인자만 모은다.

    Returns:
        list[dict]: add()에 전달된 인자가 쌓이는 목록
    """
    recorded: list[dict] = []

    @contextlib.asynccontextmanager
    async def fake_scope():
        yield object()

    class FakeRepository:
        def __init__(self, session) -> None:
            self.session = session

        async def add(self, **kwargs):
            recorded.append(kwargs)

    monkeypatch.setattr(module, "session_scope", fake_scope)
    monkeypatch.setattr(module, "DiscardedProblemRepository", FakeRepository)
    return recorded


# ── 기록할 값 ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_record_carries_the_reason_and_stage() -> None:
    record = build_discard_record(await make_state())

    assert record["reason"] == DiscardReason.MISMATCH_LABEL
    assert record["detail"] == "요청 LV2 / 생성 LV5"
    assert record["stage"] == "static_validate"


@pytest.mark.asyncio
async def test_record_converts_the_category_to_the_enum() -> None:
    """GraphState.requested_category는 str이고 컬럼은 enum이다."""
    record = build_discard_record(await make_state())

    assert record["requested_category"] is Category.HASH_TABLE


@pytest.mark.asyncio
async def test_record_keeps_the_whole_state_for_later() -> None:
    record = build_discard_record(await make_state())

    assert record["raw_payload"]["problem_title"]
    assert record["raw_payload"]["input_constraints"]


@pytest.mark.asyncio
async def test_missing_stage_falls_back_to_unknown() -> None:
    record = build_discard_record(await make_state(discard_stage=None))

    assert record["stage"] == "unknown"


@pytest.mark.asyncio
async def test_no_reason_means_nothing_to_record() -> None:
    """폐기로 왔는데 사유가 없으면 discard()를 거치지 않은 것이다."""
    assert build_discard_record(await make_state(discard_reason=None)) is None


@pytest.mark.asyncio
async def test_unknown_category_is_not_recorded() -> None:
    record = build_discard_record(await make_state(requested_category="없는카테고리"))

    assert record is None


# ── 노드 ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_node_writes_one_row(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = fix_repository(monkeypatch)

    result = await discard_problem(await make_state())

    assert len(recorded) == 1
    assert recorded[0]["reason"] == DiscardReason.MISMATCH_LABEL
    assert result == {}  # 상태를 바꾸지 않는다


@pytest.mark.asyncio
async def test_node_writes_nothing_without_a_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = fix_repository(monkeypatch)

    await discard_problem(await make_state(discard_reason=None))

    assert recorded == []


@pytest.mark.asyncio
async def test_node_survives_a_database_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """로그를 못 남긴 것 때문에 요청 전체를 실패로 만들지 않는다."""

    @contextlib.asynccontextmanager
    async def broken_scope():
        raise OSError("connection refused")
        yield  # pragma: no cover

    monkeypatch.setattr(module, "session_scope", broken_scope)

    result = await discard_problem(await make_state())

    assert result == {}


@pytest.mark.asyncio
async def test_node_survives_a_failing_insert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @contextlib.asynccontextmanager
    async def fake_scope():
        yield object()

    class BrokenRepository:
        def __init__(self, session) -> None:
            pass

        async def add(self, **kwargs):
            raise RuntimeError("컬럼 제약 위반")

    monkeypatch.setattr(module, "session_scope", fake_scope)
    monkeypatch.setattr(module, "DiscardedProblemRepository", BrokenRepository)

    assert await discard_problem(await make_state()) == {}
