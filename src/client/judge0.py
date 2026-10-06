"""judge0 코드 실행 클라이언트.

run_code는 입력 하나를 wait=true로 실행한다.
run_batch는 같은 코드를 여러 입력으로 한 번에 제출하고 결과를 폴링한다.
입력이 여러 개면 run_batch를 쓴다. wait=true 요청은 결과가 나올 때까지 연결을 붙잡는다.

샌드박스, 컴파일, 동시 실행 관리는 채점 서버가 맡는다
"""

import asyncio
import logging
import time

import httpx
from pydantic import BaseModel

from src.core.config import get_settings
from src.core.enums import ExecutionStatus, Language
from src.schema.problem import ExecutionLimit

logger = logging.getLogger(__name__)

_settings = get_settings()

JUDGE0_URL = _settings.judge0_url

# HTTP 요청 하나의 타임아웃.
# run_code는 wait=true라 큐 대기, 컴파일, 실행을 모두 포함한다.
REQUEST_TIMEOUT_S = 60.0

LANGUAGE_TO_ID = {
    Language.JAVA: 62,
    Language.CPP: 54,
    Language.PYTHON: 71,
    Language.JAVASCRIPT: 63,
}

# judge0 결과 조회 간격
POLLING_INTERVAL_S = 1.0

# 제출부터 결과를 기다리는 최대 시간
POLLING_DEADLINE_S = 120.0

# judge0 status.id 중 아직 끝나지 않은 상태. 1: 대기, 2: 실행 중
_PENDING_STATUS_IDS = {1, 2}

# batch 조회로 받아올 필드. _to_result가 읽는 필드와 토큰 매칭용 token
_RESULT_FIELDS = "token,status,stdout,stderr,compile_output,message"

# judge0 status.id -> 실행 결과. 7~12는 런타임 에러(시그널, 0이 아닌 종료 코드)다.
_STATUS = {
    3: ExecutionStatus.SUCCEEDED,
    5: ExecutionStatus.TIMED_OUT,
    6: ExecutionStatus.COMPILE_ERROR,
    **{status_id: ExecutionStatus.RUNTIME_ERROR for status_id in range(7, 13)},
}


class RunResult(BaseModel):
    """코드 한 번 실행의 결과."""

    status: ExecutionStatus
    stdout: str = ""
    stderr: str = ""

    @property
    def succeeded(self) -> bool:
        return self.status is ExecutionStatus.SUCCEEDED


def _new_client() -> httpx.AsyncClient:
    """judge0 HTTP 클라이언트. 테스트에서 가짜 전송으로 바꿔 끼운다."""
    return httpx.AsyncClient(base_url=JUDGE0_URL, timeout=REQUEST_TIMEOUT_S)


def _to_result(body: dict) -> RunResult:
    """judge0 응답 본문을 RunResult로 옮긴다."""
    status_id = (body.get("status") or {}).get("id")
    status = _STATUS.get(status_id, ExecutionStatus.INTERNAL_ERROR)
    stderr = body.get("stderr") or body.get("compile_output") or body.get("message")
    return RunResult(
        status=status, stdout=body.get("stdout") or "", stderr=stderr or ""
    )


def _build_payload(
    code: str, language: Language, stdin: str, limit: ExecutionLimit
) -> dict:
    """judge0 제출 본문 하나를 만든다."""
    return {
        "source_code": code,
        "language_id": LANGUAGE_TO_ID[language],
        "stdin": stdin,
        "cpu_time_limit": limit.time_limit_ms / 1000,
    }


async def _poll_batch(client: httpx.AsyncClient, tokens: list[str]) -> dict[str, dict]:
    """
    batch 제출 결과를 모두 끝나거나 대기 한도를 넘길 때까지 조회한다.

    한도를 넘기면 그때까지 끝난 것만 돌려준다.

    Parameters:
        client (httpx.AsyncClient): 제출에 쓴 클라이언트
        tokens (list[str]): 제출 토큰 목록

    Returns:
        dict[str, dict]: 끝난 제출의 토큰 -> judge0 응답 본문
    """
    finished: dict[str, dict] = {}
    deadline = time.monotonic() + POLLING_DEADLINE_S
    while len(finished) < len(tokens) and time.monotonic() < deadline:
        await asyncio.sleep(POLLING_INTERVAL_S)
        pending = [token for token in tokens if token not in finished]
        response = await client.get(
            "/submissions/batch",
            params={
                "tokens": ",".join(pending),
                "base64_encoded": "false",
                "fields": _RESULT_FIELDS,
            },
        )
        response.raise_for_status()
        for item in response.json()["submissions"]:
            if item and item["status"]["id"] not in _PENDING_STATUS_IDS:
                finished[item["token"]] = item
    return finished


async def run_code(
    code: str, language: Language, stdin: str, limit: ExecutionLimit
) -> RunResult:
    """
    judge0에 코드를 제출해 한 번 실행하고 결과를 돌려준다.

    예외를 올리지 않는다. 채점 서버 장애도 INTERNAL_ERROR로 알린다.

    Parameters:
        code (str): 실행할 소스 코드
        language (Language): 코드의 언어
        stdin (str): 표준 입력으로 넣을 문자열
        limit (ExecutionLimit): 실행 제한. 시간 제한만 보낸다

    Returns:
        RunResult: 실행 결과
    """
    payload = _build_payload(code, language, stdin, limit)
    try:
        async with _new_client() as client:
            response = await client.post(
                "/submissions",
                params={"wait": "true", "base64_encoded": "false"},
                json=payload,
            )
            response.raise_for_status()
            return _to_result(response.json())
    except (httpx.HTTPError, ValueError) as error:
        logger.warning("judge0 실행 실패: %r", error, exc_info=True)
        return RunResult(
            status=ExecutionStatus.INTERNAL_ERROR, stderr=f"judge0 실행 실패: {error!r}"
        )


async def run_batch(
    code: str, language: Language, stdins: list[str], limit: ExecutionLimit
) -> list[RunResult]:
    """
    judge0에 코드와 입력들을 한번에 모아서 요청하고 결과들을 돌려준다.

    예외를 올리지 않는다. 채점 서버 장애와 결과 대기 초과는 INTERNAL_ERROR로 알린다.
    한 번에 보낼 수 있는 입력 수는 judge0 MAX_SUBMISSION_BATCH_SIZE(기본 20)까지다.

    Parameters:
        code (str): 실행할 소스 코드
        language (Language): 코드의 언어
        stdins (list[str]): 입력마다 하나씩 실행할 표준 입력 목록
        limit (ExecutionLimit): 실행 제한. 시간 제한만 보낸다

    Returns:
        list[RunResult]: 입력 순서와 같은 실행 결과
    """
    if not stdins:
        return []

    submissions = [_build_payload(code, language, stdin, limit) for stdin in stdins]

    try:
        async with _new_client() as client:
            response = await client.post(
                "/submissions/batch",
                params={"base64_encoded": "false"},
                json={"submissions": submissions},
            )
            response.raise_for_status()
            tokens = [item.get("token") for item in response.json()]

            # 코드와 언어가 같으므로 하나가 거절되면 사실상 전부 거절이다
            if not all(tokens):
                raise ValueError(f"토큰을 받지 못한 제출이 있음: {response.text[:200]}")

            finished = await _poll_batch(client, tokens)
    except (httpx.HTTPError, ValueError, KeyError) as error:
        logger.warning("judge0 batch 실행 실패: %r", error, exc_info=True)

        return [
            RunResult(
                status=ExecutionStatus.INTERNAL_ERROR,
                stderr=f"judge0 batch 실행 실패: {error!r}",
            )
            for _ in stdins
        ]

    missing = len(tokens) - len(finished)
    if missing:
        logger.warning("judge0 결과 대기 시간 초과: %d/%d건", missing, len(tokens))
    return [
        _to_result(finished[token])
        if token in finished
        else RunResult(
            status=ExecutionStatus.INTERNAL_ERROR, stderr="judge0 결과 대기 시간 초과"
        )
        for token in tokens
    ]
