from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.batch.generate import run_generate
from src.batch.scheduler import MISFIRE_GRACE_S, create_scheduler, start_scheduler
from src.batch.send import run_send
from src.core.config import get_settings

SEOUL = ZoneInfo("Asia/Seoul")


def jobs() -> dict:
    return {job.id: job for job in create_scheduler().get_jobs()}


def next_fire(job, now: datetime) -> datetime:
    return job.trigger.get_next_fire_time(None, now)


@pytest.mark.parametrize(
    ("job_id", "now", "expected"),
    [
        ("batch_generate", datetime(2026, 9, 27, 0, 30), datetime(2026, 9, 27, 1, 0)),
        ("batch_generate", datetime(2026, 9, 27, 1, 0, 1), datetime(2026, 9, 28, 1, 0)),
        ("batch_send", datetime(2026, 9, 27, 1, 30), datetime(2026, 9, 27, 3, 0)),
        ("batch_send", datetime(2026, 9, 27, 4, 0), datetime(2026, 9, 28, 3, 0)),
    ],
)
def test_jobs_fire_on_the_hour_in_seoul_time(job_id, now, expected) -> None:
    job = jobs()[job_id]

    fire = next_fire(job, now.replace(tzinfo=SEOUL))

    assert fire == expected.replace(tzinfo=SEOUL)


def test_timezone_comes_from_settings(monkeypatch) -> None:
    """서버 시계가 UTC여도 설정한 시간대의 01:00에 돈다."""
    monkeypatch.setattr(get_settings(), "batch_timezone", "UTC")
    utc = ZoneInfo("UTC")

    fire = next_fire(jobs()["batch_generate"], datetime(2026, 9, 27, 0, 30, tzinfo=utc))

    assert fire == datetime(2026, 9, 27, 1, 0, tzinfo=utc)


def test_jobs_run_the_batch_entrypoints() -> None:
    registered = jobs()

    assert registered["batch_generate"].func is run_generate
    assert registered["batch_send"].func is run_send


@pytest.mark.asyncio
async def test_a_slow_night_does_not_stack_runs() -> None:
    # job_defaults는 스케줄러가 시작할 때 잡에 채워진다
    scheduler = create_scheduler()
    scheduler.start(paused=True)
    try:
        for job in scheduler.get_jobs():
            assert job.max_instances == 1
            assert job.coalesce is True
            assert job.misfire_grace_time == MISFIRE_GRACE_S
    finally:
        scheduler.shutdown(wait=False)


def test_disabled_by_default_starts_nothing() -> None:
    assert get_settings().batch_enabled is False
    assert start_scheduler() is None


@pytest.mark.asyncio
async def test_enabled_starts_the_scheduler(monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "batch_enabled", True)

    scheduler = start_scheduler()
    try:
        assert scheduler is not None
        assert scheduler.running
        assert {job.id for job in scheduler.get_jobs()} == {
            "batch_generate",
            "batch_send",
        }
    finally:
        scheduler.shutdown(wait=False)
