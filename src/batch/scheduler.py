"""새벽 배치 스케줄러.

FastAPI 프로세스 안에서 APScheduler로 돌린다. BATCH_ENABLED인 서버에서만 띄운다.
blue/green 전환 중 두 컨테이너가 함께 떠 있어도 advisory lock이 한 곳만 실행하게 한다.

종료할 때 실행 중인 배치는 취소된다. 그때까지 저장·전송한 문제는 남고
나머지는 다음 날 이어서 채운다. 잡고 있던 advisory lock은 커넥션이 닫히며 풀린다.
"""

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from src.batch.generate import run_generate
from src.batch.send import run_send
from src.core.config import get_settings

logger = logging.getLogger(__name__)

GENERATE_HOUR = 1
SEND_HOUR = 3

# 정각에 이벤트 루프가 잠깐 바빠도 건너뛰지 않게 한다. APScheduler 기본값은 1초다.
MISFIRE_GRACE_S = 600


def create_scheduler() -> AsyncIOScheduler:
    """생성(01:00)·전송(03:00) 잡을 등록한 스케줄러를 만든다. 시작은 하지 않는다."""
    timezone = get_settings().batch_timezone
    scheduler = AsyncIOScheduler(
        timezone=timezone,
        job_defaults={
            "max_instances": 1,  # 전날 배치가 아직 돌면 겹쳐 띄우지 않는다
            "coalesce": True,  # 밀린 실행이 여러 번이어도 한 번만
            "misfire_grace_time": MISFIRE_GRACE_S,
        },
    )
    scheduler.add_job(
        run_generate,
        CronTrigger(hour=GENERATE_HOUR, timezone=timezone),
        id="batch_generate",
    )
    scheduler.add_job(
        run_send,
        CronTrigger(hour=SEND_HOUR, timezone=timezone),
        id="batch_send",
    )
    return scheduler


def start_scheduler() -> AsyncIOScheduler | None:
    """
    BATCH_ENABLED일 때만 스케줄러를 시작한다. 이벤트 루프 안에서 불러야 한다.

    Returns:
        AsyncIOScheduler | None: 시작한 스케줄러. 꺼져 있으면 None
    """
    if not get_settings().batch_enabled:
        logger.info("BATCH_ENABLED가 꺼져 있어 배치 스케줄러를 띄우지 않음")
        return None

    scheduler = create_scheduler()
    scheduler.start()
    for job in scheduler.get_jobs():
        logger.info("배치 예약: %s, 다음 실행 %s", job.id, job.next_run_time)
    return scheduler
