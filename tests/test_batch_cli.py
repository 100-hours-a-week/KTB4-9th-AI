import pytest

from src.batch import __main__ as module
from src.batch.__main__ import main, parse_args


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
async def test_pool_is_closed_even_when_the_job_fails(monkeypatch) -> None:
    closed = fix_engine(monkeypatch)

    async def broken() -> None:
        raise RuntimeError("실패")

    monkeypatch.setitem(module.JOBS, "generate", broken)

    with pytest.raises(RuntimeError):
        await main("generate")
    assert closed == ["closed"]
