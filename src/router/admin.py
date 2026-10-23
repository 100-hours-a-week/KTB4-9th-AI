"""배치를 손으로 실행하는 관리자 API.

ECS에 들어가 CLI(python -m src.batch)를 치지 않아도 되게 같은 배치 함수를 연다.
인증은 X-Admin-Token 공유 토큰 하나라 누가 불렀는지 구분할 수 없으므로,
호출마다 출발지 IP를 로그로 남긴다.
"""

import logging
import secrets
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.security import APIKeyHeader

from src.batch.generate import build_plan, run_generate
from src.batch.launcher import start
from src.batch.send import run_send
from src.core.config import get_settings
from src.core.enums import BatchStep, LockKey
from src.core.exception import BatchAlreadyRunningError, UnauthorizedError
from src.db.lock import is_advisory_locked
from src.schema.base import BaseResponse
from src.schema.batch import BatchPlanResponse

logger = logging.getLogger(__name__)

admin_token_header = APIKeyHeader(name="X-Admin-Token", auto_error=False)


def _client_ip(request: Request) -> str:
    """
    요청한 곳의 IP. ALB 뒤에서는 X-Forwarded-For의 첫 번째 값이다.

    X-Forwarded-For는 클라이언트가 조작할 수 있으므로 로그에만 쓴다.
    인증이나 차단의 근거로 쓰지 않는다.

    Parameters:
        request (Request): 들어온 요청

    Returns:
        str: 출발지 IP. 알 수 없으면 "unknown"
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        # "원래IP, 프록시1, 프록시2" 순서로 쌓인다
        return forwarded.split(",")[0].strip()
    if request.client is not None:
        return request.client.host
    return "unknown"


async def require_admin_token(
    request: Request,
    token: Annotated[str | None, Depends(admin_token_header)],
) -> None:
    """
    X-Admin-Token 헤더가 서버에 설정된 관리자 토큰과 같은지 확인한다.

    서버에 토큰이 없거나, 헤더가 없거나, 값이 다르면 모두 같은 401로 거절한다.
    응답으로는 이유를 구분하지 않고, 운영자가 원인을 찾을 수 있게 로그에만 남긴다.

    Parameters:
        request (Request): 들어온 요청. 실패 로그에 출발지 IP를 남길 때 쓴다
        token (str | None): X-Admin-Token 헤더 값. 없으면 None

    Raises:
        UnauthorizedError: 인증에 실패한 경우
    """
    settings = get_settings()

    if not settings.admin_token:
        reason = "서버에 ADMIN_TOKEN이 설정되지 않음"
    elif token is None:
        reason = "X-Admin-Token 헤더 없음"
    elif not secrets.compare_digest(
        token.encode(), settings.admin_token.get_secret_value().encode()
    ):
        reason = "토큰 불일치"
    else:
        return

    # 토큰 값은 맞든 틀리든 남기지 않는다
    logger.warning("관리자 인증 실패: %s (from %s)", reason, _client_ip(request))
    raise UnauthorizedError("관리자 인증에 실패했습니다.")


router = APIRouter(prefix="/api/admin", dependencies=[Depends(require_admin_token)])


async def _launch(key: LockKey, job: Callable[[], Awaitable[None]], label: str) -> None:
    """
    배치를 백그라운드로 띄운다. 이미 돌고 있으면 409로 거절한다.

    pg_locks는 다른 프로세스(다른 컨테이너, CLI, 스케줄러)가 돌리는 배치를,
    start는 이 프로세스가 띄웠지만 아직 lock을 잡지 않은 배치를 잡아낸다.
    둘 다 안내용이고, 실제 중복 실행은 배치 안의 try_advisory_lock이 막는다.

    Parameters:
        key (LockKey): 배치가 잡을 advisory lock 키
        job (Callable[[], Awaitable[None]]): 인자를 partial로 묶은 배치 함수
        label (str): 409 메시지와 로그에 쓸 배치 이름 (예: "생성")

    Raises:
        BatchAlreadyRunningError: 같은 배치가 이미 돌고 있는 경우
    """
    if await is_advisory_locked(key) or not start(key, job):
        raise BatchAlreadyRunningError(f"{label} 배치가 이미 실행 중입니다.")
    logger.info("%s 배치를 백그라운드로 띄움", label)


@router.get("/batch/plan", tags=["Admin"])
async def get_plan(request: Request) -> BatchPlanResponse:
    """생성 배치가 지금 돈다면 무엇을 몇 개 만들지 돌려준다. 아무것도 쓰지 않는다."""
    logger.info("관리자 요청: 생성 계획 조회 (from %s)", _client_ip(request))
    plan = await build_plan()

    return BatchPlanResponse(**plan.model_dump())


@router.post("/batch/generate", status_code=202, tags=["Admin"])
async def generate_batch(
    request: Request, only: BatchStep | None = None
) -> BaseResponse:
    """
    생성 배치를 백그라운드로 띄우고 바로 202로 응답한다.

    결과는 응답으로 알 수 없고 서버 로그로 확인한다.
    only가 없으면 데일리, 배틀, 일반을 모두 만든다.
    """
    logger.info(
        "관리자 요청: 생성 배치 only=%s (from %s)", only or "all", _client_ip(request)
    )
    await _launch(LockKey.BATCH_GENERATE, partial(run_generate, only=only), "생성")
    return BaseResponse(message="생성 배치를 시작했습니다.")


@router.post("/batch/send", status_code=202, tags=["Admin"])
async def send_batch(request: Request, only: BatchStep | None = None) -> BaseResponse:
    """
    전송 배치를 백그라운드로 띄우고 바로 202로 응답한다.

    결과는 응답으로 알 수 없고 서버 로그로 확인한다.
    only가 없으면 데일리, 배틀, 일반을 모두 보낸다.
    """
    logger.info(
        "관리자 요청: 전송 배치 only=%s (from %s)", only or "all", _client_ip(request)
    )
    await _launch(LockKey.BATCH_SEND, partial(run_send, only=only), "전송")
    return BaseResponse(message="전송 배치를 시작했습니다.")
