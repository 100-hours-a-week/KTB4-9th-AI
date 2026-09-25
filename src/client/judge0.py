"""judge0 코드 실행 클라이언트.

호출자는 run_code 하나만 쓴다. 제출하고 결과가 나올 때까지 기다린다(wait=true).
샌드박스, 컴파일, 동시 실행 관리는 채점 서버가 맡는다.
"""

import logging

import httpx
from pydantic import BaseModel

from src.core.config import get_settings
from src.core.enums import ExecutionStatus, Language
from src.schema.problem import ExecutionLimit

logger = logging.getLogger(__name__)

_settings = get_settings()

JUDGE0_URL = _settings.judge0_url

# wait=true 요청 하나가 기다리는 시간. 큐 대기, 컴파일, 실행을 모두 포함한다.
REQUEST_TIMEOUT_S = 60.0

LANGUAGE_TO_ID = {
    Language.JAVA: 62,
    Language.CPP: 54,
    Language.PYTHON: 71,
    Language.JAVASCRIPT: 63,
}

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
    payload = {
        "source_code": code,
        "language_id": LANGUAGE_TO_ID[language],
        "stdin": stdin,
        "cpu_time_limit": limit.time_limit_ms / 1000,
    }
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
