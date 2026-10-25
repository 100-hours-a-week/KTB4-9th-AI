"""관리자 API가 띄운 배치를 백그라운드에서 돌리고 관리한다.

요청 안에서 기다리면 몇십 분짜리 배치가 ALB timeout에 끊기므로 태스크로 띄우고
바로 응답한다. 띄운 태스크는 여기서 참조를 쥐고 있어야 GC에 수거되지 않는다.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from functools import partial

from src.core.enums import LockKey

logger = logging.getLogger(__name__)

# 키가 advisory lock과 같아서 lock을 함께 쓰는 배치는 동시에 하나만 띄운다.
_running: dict[LockKey, asyncio.Task[None]] = {}


def start(key: LockKey, job: Callable[[], Awaitable[None]]) -> bool:
    """
    배치를 백그라운드로 띄운다.

    202를 준 뒤 태스크가 advisory lock을 잡기 전까지는 pg_locks에 보이지 않는다.
    그 사이에 온 같은 요청을 이 프로세스 안에서 막는다. 확인과 등록 사이에
    await가 없어 다른 요청이 끼어들 수 없다.

    Parameters:
        key (LockKey): 배치가 잡을 advisory lock 키. 같은 키는 동시에 하나만 돈다
        job (Callable[[], Awaitable[None]]): 인자 없이 부를 배치 함수.
            코루틴을 바로 받으면 거절할 때 실행되지 않은 코루틴이 남으므로
            띄울 때만 부른다. 인자는 functools.partial로 묶어 넘긴다

    Returns:
        bool: 띄웠으면 True. 이 프로세스에서 같은 키의 배치가 돌고 있으면 False
    """
    if key in _running:
        return False

    task = asyncio.create_task(job())
    _running[key] = task
    task.add_done_callback(partial(_on_done, key))
    return True


def _on_done(key: LockKey, task: asyncio.Task[None]) -> None:
    """끝난 배치를 목록에서 빼고, 예외로 끝났으면 남긴다."""
    _running.pop(key, None)
    if task.cancelled():
        return

    # 아무도 await하지 않으므로 여기서 꺼내지 않으면 예외가 조용히 사라진다.
    error = task.exception()
    if error is not None:
        logger.error("배치 작업 실패 (key=%s)", key.name, exc_info=error)


async def stop_all() -> None:
    """
    돌고 있는 배치를 모두 취소하고 끝날 때까지 기다린다. 앱 종료 때 부른다.

    cancel()은 요청일 뿐이라 기다려야 try_advisory_lock이 lock을 풀고
    커넥션을 반납한다. 그 전에 커넥션 풀을 닫으면 정리가 엉킨다.
    """
    # 태스크가 끝나며 _on_done이 _running을 고치므로 복사해 둔다.
    tasks = list(_running.values())
    if tasks:
        # 배포 중에 끊긴 배치가 있었는지 나중에 로그로 확인할 수 있게 남긴다
        logger.warning(
            "앱 종료로 실행 중인 배치를 취소함: %s",
            ", ".join(key.name for key in _running),
        )
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
