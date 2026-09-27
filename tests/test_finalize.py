import contextlib
import uuid

import pytest

from src.core.constants import EMBEDDING_DIM
from src.core.enums import (
    ConstraintDataType,
    ConstraintScope,
    ProblemPurpose,
    Trigger,
)
from src.problem.nodes import finalize as module
from src.problem.nodes.finalize import (
    EMBEDDING_MODEL,
    ProblemSaveError,
    build_problem,
    finalize,
)
from src.problem.state import (
    GraphState,
    HiddenTestCase,
    HintComment,
    InputConstraint,
    SolutionCode,
)
from src.schema.problem import Problem
from tests import fake_nodes

PROBLEM_ID = uuid.UUID("11111111-2222-3333-4444-555555555555")
ALGORITHM_CORE = "해시맵으로 합이 M인 쌍의 수를 센다"
VECTOR = [0.1] * EMBEDDING_DIM


async def make_state(**updates) -> GraphState:
    """검증을 모두 통과한 상태."""
    base = GraphState(requested_difficulty="LV2", requested_category="HASH_TABLE")
    state = base.model_copy(update=await fake_nodes.generate_problem(base))
    return state.model_copy(
        update={
            "is_semantically_valid": True,
            "hidden_test_cases": [HiddenTestCase(input="1 2", output="3")],
            "solution_codes": [SolutionCode(language="python", content="print(3)")],
            "hint_comments": [HintComment(language="python", content="힌트")],
            "solution_keywords": ["해시맵", "누적 카운트"],
            "algorithm_core": ALGORITHM_CORE,
            "algorithm_core_embedding": VECTOR,
            **updates,
        }
    )


def fix_repository(monkeypatch: pytest.MonkeyPatch) -> tuple[list[Problem], list[dict]]:
    """DB를 건드리지 않고 넘어온 값만 모은다.

    Returns:
        tuple[list[Problem], list[dict]]: (저장된 문제, 색인된 임베딩 인자)
    """
    saved: list[Problem] = []
    indexed: list[dict] = []

    @contextlib.asynccontextmanager
    async def fake_scope():
        yield object()

    class FakeRow:
        id = PROBLEM_ID

    class FakeProblemRepository:
        def __init__(self, session) -> None:
            pass

        async def add(self, problem: Problem, *, trigger, purpose):
            saved.append(problem)
            return FakeRow()

    class FakeEmbeddingRepository:
        def __init__(self, session) -> None:
            pass

        async def add(self, **kwargs):
            indexed.append(kwargs)

    monkeypatch.setattr(module, "session_scope", fake_scope)
    monkeypatch.setattr(module, "GeneratedProblemRepository", FakeProblemRepository)
    monkeypatch.setattr(module, "ProblemEmbeddingRepository", FakeEmbeddingRepository)
    return saved, indexed


# ── 스키마 변환 ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_every_problem_field_is_carried_over() -> None:
    state = await make_state()

    problem = build_problem(state)

    assert problem.problem_title == state.problem_title
    assert problem.problem_content == state.problem_content
    assert problem.category == state.category
    assert problem.difficulty == state.difficulty
    assert problem.requested_difficulty == state.requested_difficulty
    assert problem.category_select_reason == state.category_select_reason


@pytest.mark.asyncio
async def test_generated_artifacts_are_carried_over() -> None:
    problem = build_problem(await make_state())

    assert len(problem.hidden_test_cases) == 1
    assert problem.hidden_test_cases[0].output == "3"
    assert problem.solution_codes[0].content == "print(3)"
    assert problem.hint_comments[0].content == "힌트"
    assert problem.solution_keywords == ["해시맵", "누적 카운트"]


@pytest.mark.asyncio
async def test_numeric_constraint_range_is_carried_over() -> None:
    problem = build_problem(await make_state())
    constraints = {c.target: c for c in problem.input_constraints}

    assert constraints["N"].min_value == 2
    assert constraints["N"].max_value == 100000


@pytest.mark.asyncio
async def test_string_constraint_keeps_an_empty_range() -> None:
    """문자열 제약은 min_value, max_value가 비어 있다."""
    state = await make_state(
        input_constraints=[
            InputConstraint(
                target="S",
                scope=ConstraintScope.INPUT,
                data_type=ConstraintDataType.STRING,
            )
        ]
    )

    problem = build_problem(state)

    assert problem.input_constraints[0].min_value is None
    assert problem.input_constraints[0].max_value is None


@pytest.mark.asyncio
async def test_graph_only_fields_are_dropped() -> None:
    problem = build_problem(await make_state())

    assert not hasattr(problem, "is_discarded")
    assert not hasattr(problem, "reference_code")


@pytest.mark.asyncio
async def test_trigger_and_purpose_are_not_sent_to_spring() -> None:
    """Problem은 Spring으로 나가는 payload다. 관리용 값이 섞이면 안 된다."""
    state = await make_state(trigger=Trigger.BATCH, purpose=ProblemPurpose.DAILY)

    payload = build_problem(state).model_dump(mode="json", by_alias=True)

    assert "trigger" not in payload
    assert "purpose" not in payload


@pytest.mark.asyncio
async def test_missing_required_field_is_an_error() -> None:
    state = await make_state(difficulty=None)

    with pytest.raises(ProblemSaveError):
        build_problem(state)


# ── 노드 ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_node_saves_the_problem_and_returns_its_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved, indexed = fix_repository(monkeypatch)

    result = await finalize(await make_state())

    assert len(saved) == 1
    assert saved[0].problem_title
    assert result == {"problem_id": PROBLEM_ID}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("trigger", "purpose"),
    [
        (Trigger.ON_DEMAND, ProblemPurpose.NORMAL),
        (Trigger.BATCH, ProblemPurpose.DAILY),
    ],
)
async def test_node_saves_trigger_and_purpose_from_state(
    monkeypatch: pytest.MonkeyPatch, trigger: Trigger, purpose: ProblemPurpose
) -> None:
    """이 값이 틀리면 새벽 전송에서 빠지거나 엉뚱한 엔드포인트로 나간다."""
    options: list[dict] = []

    @contextlib.asynccontextmanager
    async def fake_scope():
        yield object()

    class FakeRow:
        id = PROBLEM_ID

    class RecordingRepository:
        def __init__(self, session) -> None:
            pass

        async def add(self, problem, **kwargs):
            options.append(kwargs)
            return FakeRow()

    async def skip_index(*args) -> None:
        pass

    monkeypatch.setattr(module, "session_scope", fake_scope)
    monkeypatch.setattr(module, "GeneratedProblemRepository", RecordingRepository)
    monkeypatch.setattr(module, "index_embedding", skip_index)

    await finalize(await make_state(trigger=trigger, purpose=purpose))

    assert options == [{"trigger": trigger, "purpose": purpose}]


@pytest.mark.asyncio
async def test_save_failure_is_raised_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """저장에 실패하면 만들어 낸 문제가 사라진다. 성공으로 위장해선 안 된다."""

    @contextlib.asynccontextmanager
    async def broken_scope():
        raise OSError("connection refused")
        yield  # pragma: no cover

    monkeypatch.setattr(module, "session_scope", broken_scope)

    with pytest.raises(ProblemSaveError):
        await finalize(await make_state())


@pytest.mark.asyncio
async def test_failing_insert_is_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    @contextlib.asynccontextmanager
    async def fake_scope():
        yield object()

    class BrokenRepository:
        def __init__(self, session) -> None:
            pass

        async def add(self, problem, **options):
            raise RuntimeError("컬럼 제약 위반")

    monkeypatch.setattr(module, "session_scope", fake_scope)
    monkeypatch.setattr(module, "GeneratedProblemRepository", BrokenRepository)

    with pytest.raises(ProblemSaveError):
        await finalize(await make_state())


# ── 임베딩 색인 ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_embedding_is_indexed_with_the_problem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """색인이 없으면 다음 문제의 중복 검사가 이 문제를 못 찾는다."""
    _, indexed = fix_repository(monkeypatch)

    await finalize(await make_state())

    assert len(indexed) == 1
    assert indexed[0]["problem_id"] == PROBLEM_ID
    assert indexed[0]["embedding"] == VECTOR
    assert indexed[0]["algorithm_core"] == ALGORITHM_CORE
    assert indexed[0]["embedding_model"] == EMBEDDING_MODEL


@pytest.mark.asyncio
async def test_index_reuses_the_embedding_from_check_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """finalize는 임베딩을 다시 만들지 않는다."""
    _, indexed = fix_repository(monkeypatch)

    await finalize(await make_state())

    assert indexed[0]["embedding"] is VECTOR


@pytest.mark.asyncio
async def test_problem_is_still_saved_without_an_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """색인을 못 해도 만들어 낸 문제를 버리지 않는다."""
    saved, indexed = fix_repository(monkeypatch)

    result = await finalize(await make_state(algorithm_core_embedding=None))

    assert len(saved) == 1
    assert indexed == []
    assert result == {"problem_id": PROBLEM_ID}
