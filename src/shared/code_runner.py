"""코드 실행기.

호출자는 run_code 하나만 쓴다. 지금은 로컬 subprocess로 파이썬만 실행한다.

TODO(judge0): judge0가 붙으면 이 파일의 구현만 갈아 끼운다.
  run_code의 시그니처와 RunResult는 judge0 응답을 담을 수 있게 맞춰 두었다.
  (stdout, stderr, exit_code, 상태, 소요 시간)
  샌드박스, 언어별 컴파일, 정확한 메모리 계측은 judge0가 맡는다.

실행하는 코드는 LLM이 쓴 것이다. 신뢰할 수 없다고 보고 다룬다.
- 환경변수를 물려주지 않는다. API 키와 DB 비밀번호가 새면 안 된다.
- 임시 디렉터리에서 돌린다.
- 시간 제한을 걸고, 넘으면 죽인다.
- 메모리 상한을 건다. 리눅스에서만 걸린다(MEMORY_LIMIT_SUPPORTED).

남아 있는 한계: 출력 상한은 프로세스가 끝난 뒤에 자른다. 제한 시간 안에
많이 출력하는 코드는 그만큼을 메모리에 담는다. 정확한 차단은 judge0 몫이다.
"""

import asyncio
import contextlib
import logging
import resource
import sys
import tempfile
import time
import weakref
from pathlib import Path

from pydantic import BaseModel

from src.enum import ExecutionStatus, Language
from src.problem.state import ExecutionLimit

logger = logging.getLogger(__name__)

# 이 실행기가 돌릴 수 있는 언어
SUPPORTED_LANGUAGES = frozenset({Language.PYTHON})

# macOS는 RLIMIT_AS의 하드 상한이 무한이라 낮춰 잡는 것 자체를 거부한다
# ("current limit exceeds maximum limit"). 그래서 로컬에서는 메모리 상한이
# 걸리지 않고 시간 제한만 남는다. 배포 환경(리눅스)에서는 걸린다.
MEMORY_LIMIT_SUPPORTED = sys.platform.startswith("linux")

_warned_about_memory = False

# 인터프리터가 뜨는 시간은 알고리즘 시간이 아니므로 제한에 얹어 준다
STARTUP_MARGIN_MS = 500

# stdout, stderr 각각의 상한
MAX_OUTPUT_BYTES = 64 * 1024

# 동시에 띄울 프로세스 수. 테스트 케이스를 한꺼번에 돌리면
# 각 프로세스가 메모리 상한까지 쓸 수 있어 컨테이너가 죽는다.
MAX_CONCURRENT_RUNS = 4

# Semaphore는 한 번 대기하면 그 이벤트 루프에 묶인다. 모듈 전역에 하나만 두면
# asyncio.run을 두 번 부르는 배치 러너에서 두 번째 호출이 RuntimeError로 죽는다.
# 그래서 루프마다 따로 만든다. 루프가 사라지면 항목도 같이 사라진다.
_slots: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
    weakref.WeakKeyDictionary()
)


def _slot() -> asyncio.Semaphore:
    """지금 도는 이벤트 루프의 실행 슬롯."""
    loop = asyncio.get_running_loop()
    semaphore = _slots.get(loop)
    if semaphore is None:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_RUNS)
        _slots[loop] = semaphore
    return semaphore


SOURCE_FILENAME = {Language.PYTHON: "main.py"}


class RunResult(BaseModel):
    """코드 한 번 실행의 결과."""

    status: ExecutionStatus
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_ms: int = 0

    @property
    def succeeded(self) -> bool:
        return self.status is ExecutionStatus.SUCCEEDED


def _limit_setter(limit: ExecutionLimit):
    """자식 프로세스에서 자원 상한을 건다.

    fork 직후에 도는 함수다. 예외를 올리면 프로세스 시작이 실패하므로
    상한을 걸지 못해도 그대로 진행한다. 시간 제한은 부모가 따로 건다.
    여기서 로깅하면 fork 뒤 락 상태에 따라 멈출 수 있어 로그도 남기지 않는다.
    """

    def apply() -> None:
        if MEMORY_LIMIT_SUPPORTED:
            with contextlib.suppress(OSError, ValueError):
                memory_bytes = limit.memory_limit_kb * 1024
                resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        with contextlib.suppress(OSError, ValueError):
            seconds = limit.time_limit_ms // 1000 + 1
            resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds))
        with contextlib.suppress(OSError, ValueError, AttributeError):
            resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))  # fork 금지

    return apply


def _warn_once_about_memory() -> None:
    """메모리 상한이 걸리지 않는 플랫폼임을 한 번만 알린다."""
    global _warned_about_memory
    if MEMORY_LIMIT_SUPPORTED or _warned_about_memory:
        return
    _warned_about_memory = True
    logger.warning(
        "%s에서는 메모리 상한을 걸 수 없다. 시간 제한만 적용된다.", sys.platform
    )


def _decode(raw: bytes) -> tuple[str, bool]:
    """바이트를 문자열로 만들고 상한을 넘었는지 알려준다."""
    exceeded = len(raw) > MAX_OUTPUT_BYTES
    return raw[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace"), exceeded


def _classify(
    exit_code: int | None, stderr: str, output_exceeded: bool
) -> ExecutionStatus:
    """종료 코드와 stderr로 실행 결과를 가른다.

    메모리 초과는 정확히 알 수 없다. 파이썬이 RLIMIT_AS에 걸리면
    MemoryError를 내므로 그것만 알아본다. 정확한 계측은 judge0 몫이다.
    """
    if output_exceeded:
        return ExecutionStatus.OUTPUT_EXCEEDED
    if exit_code == 0:
        return ExecutionStatus.SUCCEEDED
    if "MemoryError" in stderr:
        return ExecutionStatus.MEMORY_EXCEEDED
    return ExecutionStatus.RUNTIME_ERROR


async def run_code(
    code: str, language: Language, stdin: str, limit: ExecutionLimit
) -> RunResult:
    """
    코드를 한 번 실행하고 결과를 돌려준다.

    예외를 올리지 않는다. 실패도 RunResult의 status로 알린다.
    호출자가 여러 입력을 asyncio.gather로 한꺼번에 넘겨도 된다.
    실제로 동시에 도는 프로세스 수는 MAX_CONCURRENT_RUNS로 묶는다.

    Parameters:
        code (str): 실행할 소스 코드
        language (Language): 코드의 언어
        stdin (str): 표준 입력으로 넣을 문자열
        limit (ExecutionLimit): 시간·메모리 제한

    Returns:
        RunResult: 실행 결과
    """
    if language not in SUPPORTED_LANGUAGES:
        return RunResult(status=ExecutionStatus.UNSUPPORTED_LANGUAGE)
    _warn_once_about_memory()
    if not code.strip():
        return RunResult(status=ExecutionStatus.INTERNAL_ERROR, stderr="코드가 비었음")

    async with _slot():
        with tempfile.TemporaryDirectory(prefix="cosmos-run-") as workdir:
            source = Path(workdir) / SOURCE_FILENAME[language]
            source.write_text(code, encoding="utf-8")
            return await _run_python(source, workdir, stdin, limit)


async def _run_python(
    source: Path, workdir: str, stdin: str, limit: ExecutionLimit
) -> RunResult:
    started = time.perf_counter()
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(source),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
            # 환경변수를 물려주지 않는다. 인코딩만 정해 출력이 흔들리지 않게 한다.
            env={"PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"},
            preexec_fn=_limit_setter(limit),
        )
    except OSError as error:
        logger.warning("실행기 시작 실패: %s", error, exc_info=True)
        return RunResult(status=ExecutionStatus.INTERNAL_ERROR, stderr=str(error))

    timeout = (limit.time_limit_ms + STARTUP_MARGIN_MS) / 1000
    try:
        raw_stdout, raw_stderr = await asyncio.wait_for(
            process.communicate(stdin.encode("utf-8")), timeout=timeout
        )
    except TimeoutError:
        process.kill()
        with contextlib.suppress(Exception):
            await process.communicate()
        return RunResult(
            status=ExecutionStatus.TIMED_OUT,
            duration_ms=_elapsed_ms(started),
        )

    stdout, stdout_exceeded = _decode(raw_stdout)
    stderr, stderr_exceeded = _decode(raw_stderr)
    return RunResult(
        status=_classify(
            process.returncode, stderr, stdout_exceeded or stderr_exceeded
        ),
        stdout=stdout,
        stderr=stderr,
        exit_code=process.returncode,
        duration_ms=_elapsed_ms(started),
    )


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
