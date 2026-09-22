import asyncio
import time

import pytest

from src.enum import ExecutionStatus, Language
from src.problem.state import ExecutionLimit
from src.shared import code_runner
from src.shared.code_runner import (
    MAX_CONCURRENT_RUNS,
    MAX_OUTPUT_BYTES,
    MEMORY_LIMIT_SUPPORTED,
    SUPPORTED_LANGUAGES,
    RunResult,
    run_code,
)


def limit(time_limit_ms: int = 2000, memory_limit_kb: int = 262144) -> ExecutionLimit:
    return ExecutionLimit(
        language=Language.PYTHON,
        time_limit_ms=time_limit_ms,
        memory_limit_kb=memory_limit_kb,
    )


async def run(code: str, stdin: str = "", **kwargs) -> RunResult:
    return await run_code(code, Language.PYTHON, stdin, limit(**kwargs))


# ── 정상 실행 ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reads_stdin_and_returns_stdout() -> None:
    result = await run("n, m = map(int, input().split())\nprint(n + m)", "5 6")

    assert result.status is ExecutionStatus.SUCCEEDED
    assert result.succeeded
    assert result.stdout.strip() == "11"
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_multiline_stdin_is_delivered() -> None:
    code = "import sys\nprint(sum(int(line) for line in sys.stdin))"

    result = await run(code, "1\n2\n3\n")

    assert result.stdout.strip() == "6"


@pytest.mark.asyncio
async def test_utf8_output_survives() -> None:
    result = await run("print('한글 출력')")

    assert result.stdout.strip() == "한글 출력"


# ── 실패 ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_nonzero_exit_is_a_runtime_error() -> None:
    result = await run("raise ValueError('터짐')")

    assert result.status is ExecutionStatus.RUNTIME_ERROR
    assert result.exit_code != 0
    assert "ValueError" in result.stderr


@pytest.mark.asyncio
async def test_infinite_loop_times_out() -> None:
    result = await run("while True: pass", time_limit_ms=200)

    assert result.status is ExecutionStatus.TIMED_OUT
    assert not result.succeeded


@pytest.mark.asyncio
async def test_timeout_allows_for_interpreter_startup() -> None:
    """제한이 짧아도 인터프리터가 뜨는 시간 때문에 죽어선 안 된다."""
    result = await run("print(1)", time_limit_ms=1)

    assert result.status is ExecutionStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_syntax_error_is_reported_not_raised() -> None:
    result = await run("def broken(:\n    pass")

    assert result.status is ExecutionStatus.RUNTIME_ERROR
    assert "SyntaxError" in result.stderr


@pytest.mark.asyncio
async def test_oversized_output_is_flagged_and_cut() -> None:
    result = await run(f"print('x' * {MAX_OUTPUT_BYTES * 2})")

    assert result.status is ExecutionStatus.OUTPUT_EXCEEDED
    assert len(result.stdout.encode()) <= MAX_OUTPUT_BYTES


@pytest.mark.asyncio
async def test_unsupported_language_is_rejected_without_running() -> None:
    result = await run_code("class Main {}", Language.JAVA, "", limit())

    assert result.status is ExecutionStatus.UNSUPPORTED_LANGUAGE
    assert Language.JAVA not in SUPPORTED_LANGUAGES


@pytest.mark.asyncio
async def test_blank_code_is_rejected() -> None:
    result = await run("   \n  ")

    assert result.status is ExecutionStatus.INTERNAL_ERROR


@pytest.mark.asyncio
async def test_run_code_never_raises() -> None:
    """호출자는 예외를 잡지 않는다. 실패도 status로 온다."""
    for code in ["raise SystemExit(3)", "import os\nos._exit(9)", "while True: pass"]:
        result = await run(code, time_limit_ms=200)

        assert isinstance(result, RunResult)


# ── 격리 ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_environment_variables_are_not_inherited() -> None:
    """LLM이 쓴 코드가 API 키와 DB 비밀번호를 읽어선 안 된다."""
    code = (
        "import os\n"
        "leaked = [k for k in os.environ if 'KEY' in k or 'PASSWORD' in k]\n"
        "print(leaked)"
    )

    result = await run(code)

    assert result.stdout.strip() == "[]"


@pytest.mark.asyncio
async def test_runs_outside_the_project_directory() -> None:
    """프로젝트 파일을 건드릴 수 있는 자리에서 돌리지 않는다."""
    code = "import pathlib\nprint(sorted(p.name for p in pathlib.Path().iterdir()))"

    result = await run(code)

    assert "src" not in result.stdout
    assert "main.py" in result.stdout  # 임시 디렉터리에는 소스만 있다


@pytest.mark.asyncio
@pytest.mark.skipif(not MEMORY_LIMIT_SUPPORTED, reason="리눅스에서만 상한이 걸린다")
async def test_memory_limit_is_enforced_on_linux() -> None:
    result = await run("x = bytearray(500 * 1024 * 1024)", memory_limit_kb=32768)

    assert result.status is ExecutionStatus.MEMORY_EXCEEDED


@pytest.mark.asyncio
async def test_memory_warning_is_logged_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """상한이 걸리지 않는 플랫폼임을 조용히 넘기지 않는다."""
    monkeypatch.setattr(code_runner, "MEMORY_LIMIT_SUPPORTED", False)
    monkeypatch.setattr(code_runner, "_warned_about_memory", False)
    warnings: list[str] = []
    monkeypatch.setattr(
        code_runner.logger, "warning", lambda *args, **kw: warnings.append(args[0])
    )

    await run("print(1)")
    await run("print(2)")

    assert len(warnings) == 1


# ── 동시 실행 ──────────────────────────────────────────────────────────


def test_survives_a_new_event_loop() -> None:
    """배치 러너는 asyncio.run을 여러 번 부른다.

    실행 슬롯을 모듈 전역 Semaphore 하나로 두면 첫 루프에 묶여
    두 번째 asyncio.run이 RuntimeError로 죽는다.
    """

    async def many() -> list[RunResult]:
        return await asyncio.gather(
            *(run("print(1)") for _ in range(MAX_CONCURRENT_RUNS * 2))
        )

    for _ in range(2):
        results = asyncio.run(many())

        assert all(result.succeeded for result in results)


@pytest.mark.asyncio
async def test_many_inputs_run_concurrently() -> None:
    """20개 입력을 gather로 돌릴 수 있어야 한다. 순차로는 너무 느리다."""
    code = "import time\ntime.sleep(0.2)\nprint(input())"
    started = time.perf_counter()

    results = await asyncio.gather(
        *(run(code, str(index)) for index in range(20)),
    )
    elapsed = time.perf_counter() - started

    assert [result.stdout.strip() for result in results] == [
        str(index) for index in range(20)
    ]
    assert elapsed < 2.0  # 순차라면 4초가 넘는다
