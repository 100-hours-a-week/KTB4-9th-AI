import pytest

from src.batch import __main__ as module
from src.batch.__main__ import main, parse_args
from src.batch.generate import GENERATE_STEPS
from src.batch.send import SEND_STEPS


def fix_engine(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    closed: list[str] = []

    async def close_engine() -> None:
        closed.append("closed")

    monkeypatch.setattr(module, "close_engine", close_engine)
    return closed


@pytest.mark.parametrize("job", ["plan", "generate", "send"])
def test_known_jobs_are_accepted(job) -> None:
    assert parse_args([job]).job == job


def test_unknown_job_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_args(["deploy"])


def test_generate_and_send_have_the_same_steps() -> None:
    """--only 선택지는 한 목록이다. 이름이 어긋나면 고를 수 없는 단계가 생긴다."""
    assert list(GENERATE_STEPS) == list(SEND_STEPS)


@pytest.mark.parametrize("job", ["generate", "send"])
@pytest.mark.parametrize("step", ["daily", "battle", "normal"])
def test_only_picks_one_step(job, step) -> None:
    assert parse_args([job, "--only", step]).only == step


def test_only_defaults_to_every_step() -> None:
    assert parse_args(["send"]).only is None


def test_unknown_step_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_args(["send", "--only", "weekly"])


def test_plan_does_not_take_only() -> None:
    with pytest.raises(SystemExit):
        parse_args(["plan", "--only", "battle"])


@pytest.mark.asyncio
async def test_runs_the_chosen_job_and_closes_the_pool(monkeypatch) -> None:
    ran: list[str] = []
    closed = fix_engine(monkeypatch)

    async def fake_send() -> None:
        ran.append("send")

    monkeypatch.setitem(module.JOBS, "send", fake_send)

    await main("send")

    assert ran == ["send"]
    assert closed == ["closed"]


@pytest.mark.asyncio
async def test_only_is_passed_to_the_job(monkeypatch) -> None:
    fix_engine(monkeypatch)
    received: list[str | None] = []

    async def fake_generate(only: str | None = None) -> None:
        received.append(only)

    monkeypatch.setitem(module.JOBS, "generate", fake_generate)

    await main("generate", "battle")

    assert received == ["battle"]


@pytest.mark.asyncio
async def test_pool_is_closed_even_when_the_job_fails(monkeypatch) -> None:
    closed = fix_engine(monkeypatch)

    async def broken() -> None:
        raise RuntimeError("실패")

    monkeypatch.setitem(module.JOBS, "generate", broken)

    with pytest.raises(RuntimeError):
        await main("generate")
    assert closed == ["closed"]
