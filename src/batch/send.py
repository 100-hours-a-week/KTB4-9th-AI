"""새벽 문제 전송 배치.

03:00에 돈다. 생성 배치가 버퍼에 쌓아 둔 문제를 Spring으로 보낸다.
데일리를 먼저 보내고, 일반 문제는 청크로 나눠 보낸다.

DB 트랜잭션을 연 채로 HTTP를 보내지 않는다. 일시 장애면 20분 넘게
재시도할 수 있어서, 청크를 짧게 읽고 → 보내고 → 성공하자마자 표시한다.
"""

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable, Collection

import httpx

from src.client.backend import save_daily_problems, save_problems
from src.core.enums import LockKey, ProblemPurpose
from src.db.lock import try_advisory_lock
from src.db.repository import GeneratedProblemRepository
from src.db.session import session_scope
from src.problem.daily import DAILY_COUNT
from src.schema.problem import Problem

logger = logging.getLogger(__name__)

SEND_CHUNK_SIZE = 20

# 일시 장애 재시도 간격(초). Spring 배포·재시작을 버틴다. 첫 시도까지 총 4번.
SEND_RETRY_DELAYS_S = (60, 300, 900)

# 4xx지만 내용 문제가 아니라 기다리면 되는 응답
TRANSIENT_CLIENT_ERRORS = {408, 429}

type Chunk = list[tuple[uuid.UUID, Problem]]
type Sender = Callable[[list[Problem]], Awaitable[None]]


def is_rejection(error: Exception) -> bool:
    """Spring이 내용 때문에 거부했는지. 다시 보내도 같은 결과가 나온다."""
    if not isinstance(error, httpx.HTTPStatusError):
        return False
    status = error.response.status_code
    return 400 <= status < 500 and status not in TRANSIENT_CLIENT_ERRORS


def _reason(error: httpx.HTTPError) -> str:
    """로그용 실패 사유. 거부면 Spring이 준 본문 앞부분까지 남긴다."""
    if isinstance(error, httpx.HTTPStatusError):
        return f"{error.response.status_code} {error.response.text[:300]}"
    return repr(error)


async def send_with_retry(send: Sender, problems: list[Problem]) -> None:
    """
    보낸다. 일시 장애면 SEND_RETRY_DELAYS_S 간격으로 다시 보낸다.

    Raises:
        httpx.HTTPError: 거부된 경우(바로), 재시도해도 실패한 경우
    """
    for attempt, delay in enumerate((*SEND_RETRY_DELAYS_S, None), start=1):
        try:
            await send(problems)
            return
        except httpx.HTTPError as error:
            if is_rejection(error) or delay is None:
                raise
            logger.warning(
                "Spring 전송 실패 (%d번째), %d초 뒤 재시도: %s",
                attempt,
                delay,
                _reason(error),
            )
            await asyncio.sleep(delay)


async def load_chunk(
    purpose: ProblemPurpose, limit: int, exclude_ids: Collection[uuid.UUID] = ()
) -> Chunk:
    """보낼 문제를 오래된 순으로 읽는다. 트랜잭션은 읽는 동안만 연다."""
    async with session_scope() as session:
        repo = GeneratedProblemRepository(session)
        rows = await repo.list_pending(purpose, limit, exclude_ids)
        return [(row.id, repo.to_problem(row)) for row in rows]


async def mark_sent(problem_ids: list[uuid.UUID]) -> None:
    async with session_scope() as session:
        await GeneratedProblemRepository(session).mark_sent(problem_ids)


async def deliver(send: Sender, chunk: Chunk) -> None:
    """
    보내고, 성공하자마자 전송 완료로 표시한다.

    Raises:
        httpx.HTTPError: 전송 실패. 표시하지 않는다
        Exception: 전송은 됐지만 표시에 실패한 경우
    """
    ids = [problem_id for problem_id, _ in chunk]
    await send_with_retry(send, [problem for _, problem in chunk])
    try:
        await mark_sent(ids)
    except Exception:
        logger.error(
            "Spring 저장은 됐지만 전송 표시에 실패함. 다음 날 중복 전송될 수 있음: %s",
            [str(problem_id) for problem_id in ids],
        )
        raise


async def deliver_one_by_one(chunk: Chunk) -> set[uuid.UUID]:
    """
    거부된 청크를 한 건씩 다시 보낸다. 문제 있는 행만 골라내기 위해서다.

    Returns:
        set[uuid.UUID]: 이번에도 거부된 행

    Raises:
        httpx.HTTPError: 재시도해도 안 된 일시 장애. 앞서 보낸 건 이미 표시됐다
    """
    rejected: set[uuid.UUID] = set()
    for problem_id, problem in chunk:
        try:
            await deliver(save_problems, [(problem_id, problem)])
        except httpx.HTTPError as error:
            if not is_rejection(error):
                raise
            rejected.add(problem_id)
            logger.error(
                "Spring이 문제를 거부함 (id=%s, %s/%s): %s",
                problem_id,
                problem.category.value,
                problem.difficulty.value,
                _reason(error),
            )
    return rejected


async def send_chunk(chunk: Chunk, rejected: set[uuid.UUID]) -> int:
    """
    일반 문제 청크 하나를 보낸다. 거부되면 한 건씩 다시 보낸다.

    Parameters:
        chunk (Chunk): 보낼 문제
        rejected (set[uuid.UUID]): 거부된 행을 여기에 더한다

    Returns:
        int: 보낸 개수
    """
    try:
        await deliver(save_problems, chunk)
        return len(chunk)
    except httpx.HTTPError as error:
        if not is_rejection(error):
            raise
        logger.warning(
            "청크 %d개가 거부돼 한 건씩 다시 보냄: %s", len(chunk), _reason(error)
        )

    bad = await deliver_one_by_one(chunk)
    rejected |= bad
    return len(chunk) - len(bad)


async def send_daily() -> None:
    """
    오래된 데일리를 DAILY_COUNT개까지 한 요청으로 보낸다.

    데일리는 세트라 나눠 보내지 않는다. 실패하면 통째로 남고,
    다음 날 생성은 재고가 있어 건너뛰므로 다음 날 3시에 이 세트가 나간다.
    """
    chunk = await load_chunk(ProblemPurpose.DAILY, DAILY_COUNT)
    if not chunk:
        logger.warning("보낼 데일리가 없음")
        return
    if len(chunk) < DAILY_COUNT:
        logger.warning("데일리가 %d/%d개뿐이라 있는 만큼 보냄", len(chunk), DAILY_COUNT)

    try:
        await deliver(save_daily_problems, chunk)
    except httpx.HTTPError as error:
        # 거부든 장애든 세트째 남긴다. 사유와 id를 한 줄로 남겨 바로 찾을 수 있게 한다.
        logger.error(
            "데일리 전송 실패(%s). 세트째 남기고 다음 날 다시 보낸다: %s, ids=%s",
            "거부" if is_rejection(error) else "장애 지속",
            _reason(error),
            [str(problem_id) for problem_id, _ in chunk],
        )
        return
    logger.info("데일리 전송 끝: %d개", len(chunk))


async def send_normal() -> None:
    """미전송 일반 문제가 없을 때까지 청크로 나눠 보낸다."""
    sent = 0
    rejected: set[uuid.UUID] = set()
    try:
        while chunk := await load_chunk(
            ProblemPurpose.NORMAL, SEND_CHUNK_SIZE, rejected
        ):
            sent += await send_chunk(chunk, rejected)
    except httpx.HTTPError as error:
        # 거부는 send_chunk가 처리한다. 여기까지 온 건 재시도해도 안 된 일시 장애다.
        logger.error(
            "Spring 장애가 계속돼 일반 문제 전송을 멈춤. 남은 건 다음 날 보낸다: %s",
            _reason(error),
        )

    logger.info("일반 문제 전송 끝: %d개", sent)
    if rejected:
        logger.error(
            "Spring이 거부한 문제 %d개. 다음 날 다시 시도된다: %s",
            len(rejected),
            sorted(str(problem_id) for problem_id in rejected),
        )


async def run_send() -> None:
    """
    03:00 전송 배치. 스케줄러가 부른다.

    예외를 올리지 않는다. 스케줄러는 결과를 보지 않으므로 로그로 남긴다.
    데일리가 실패해도 일반 문제는 보낸다.
    """
    try:
        async with try_advisory_lock(LockKey.BATCH_SEND) as acquired:
            if not acquired:
                logger.info("다른 인스턴스가 전송 배치를 실행 중이라 건너뜀")
                return

            for step in (send_daily, send_normal):
                try:
                    await step()
                except Exception:
                    logger.exception("전송 배치 단계 실패: %s", step.__name__)
    except Exception:
        # 락을 잡는 DB 연결부터 실패한 경우
        logger.exception("전송 배치를 시작하지 못함")
