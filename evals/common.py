"""생성·평가 스크립트가 함께 쓰는 것.

결과 파일, 이어서 돌기, meta, LangSmith 설정, 동시 실행.

결과 파일은 요청 1건 = 1줄인 JSONL이고 덧붙이기만 한다.
같은 요청을 다시 돌리면 뒤에 한 줄이 더 생기고, 읽을 때 마지막 줄을 쓴다.
"""

import asyncio
import hashlib
import json
import os
import re
from collections.abc import Awaitable, Callable
from pathlib import Path

import langsmith.utils
from pydantic import BaseModel

from evals.requests import EvalRequest

DEFAULT_REQUESTS = Path("evals/requests_v1.jsonl")
RESULTS_DIR = Path("evals/results")
LANGSMITH_PROJECT = "cosmos-model-eval"
EVAL_TAG = "eval"
GEMINI = "gemini"  # generate_problem을 바꾸지 않았을 때 결과에 남는 모델 이름
LABEL_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


class Row(BaseModel):
    """결과 파일 한 줄의 공통 필드."""

    request_id: str


def load_rows[R: Row](path: Path, row_type: type[R]) -> dict[str, R]:
    """
    결과 파일을 읽는다. 같은 요청이 여러 줄이면 마지막 줄을 쓴다.

    Parameters:
        path (Path): 결과 JSONL
        row_type (type[R]): 한 줄의 형식

    Returns:
        dict[str, R]: 요청 id → 결과. 파일이 없으면 빈 dict
    """
    if not path.exists():
        return {}
    rows: dict[str, R] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                row = row_type.model_validate_json(line)
                rows[row.request_id] = row
    return rows


def append_row(path: Path, row: Row) -> None:
    """결과 한 줄을 바로 파일에 덧붙인다. 중간에 멈춰도 끝난 요청은 남는다."""
    with path.open("a", encoding="utf-8") as file:
        file.write(row.model_dump_json() + "\n")


def select_pending[R: Row](
    requests: list[EvalRequest],
    done: dict[str, R],
    retryable: Callable[[R], bool] | None = None,
) -> list[EvalRequest]:
    """
    아직 안 돈 요청을 고른다. retryable이 참인 결과도 다시 돈다.

    Parameters:
        requests (list[EvalRequest]): 전체 요청
        done (dict[str, R]): 이미 끝난 결과
        retryable (Callable[[R], bool] | None): 다시 돌릴 결과인지. None이면 재시도 없음

    Returns:
        list[EvalRequest]: 이번에 돌릴 요청 (원래 순서)
    """
    pending = []
    for request in requests:
        row = done.get(request.id)
        if row is None or (retryable is not None and retryable(row)):
            pending.append(request)
    return pending


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_meta(path: Path, meta: dict, identity: tuple[str, ...]) -> None:
    """
    같은 결과 파일의 이전 실행과 조건이 같은지 확인한다. 처음이면 meta를 쓴다.

    Parameters:
        path (Path): meta JSON 파일
        meta (dict): 이번 실행 조건
        identity (tuple[str, ...]): 달라지면 결과가 섞이는 키

    Raises:
        SystemExit: identity 중 하나라도 다른 경우
    """
    if not path.exists():
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        return
    previous = json.loads(path.read_text())
    changed = [key for key in identity if previous.get(key) != meta.get(key)]
    if changed:
        details = ", ".join(
            f"{k}: {previous.get(k)!r} → {meta.get(k)!r}" for k in changed
        )
        raise SystemExit(
            f"이 label은 다른 조건으로 이미 돌았습니다 ({details}). 새 label을 쓰세요."
        )


def configure_langsmith(project: str) -> None:
    """트레이스를 평가 전용 프로젝트로 보낸다.

    langsmith는 프로젝트 이름을 캐시하므로 환경변수를 바꾼 뒤 캐시를 비운다.
    """
    os.environ["LANGSMITH_PROJECT"] = project
    langsmith.utils.get_env_var.cache_clear()
    langsmith.utils.get_tracer_project.cache_clear()


async def run_concurrently[R: Row](
    requests: list[EvalRequest],
    run_one: Callable[[EvalRequest], Awaitable[R]],
    *,
    rows_path: Path,
    concurrency: int,
    describe: Callable[[R], str],
) -> list[R]:
    """
    요청을 동시에 돌리며, 끝나는 대로 결과 파일에 한 줄씩 덧붙이고 진행 상황을 출력한다.

    Parameters:
        requests (list[EvalRequest]): 이번에 돌릴 요청
        run_one (Callable): 요청 하나를 결과 한 줄로 만드는 함수. 예외를 올리지 않는다
        rows_path (Path): 결과 JSONL
        concurrency (int): 동시에 돌릴 수
        describe (Callable[[R], str]): 진행 줄에 붙일 결과 요약

    Returns:
        list[R]: 이번에 만든 결과 (끝난 순서)
    """
    semaphore = asyncio.Semaphore(concurrency)
    rows: list[R] = []

    async def one(request: EvalRequest) -> None:
        async with semaphore:
            row = await run_one(request)
        append_row(rows_path, row)
        rows.append(row)
        print(
            f"[{len(rows)}/{len(requests)}] {request.id} "
            f"{request.difficulty.value}/{request.category.value} {describe(row)}",
            flush=True,
        )

    await asyncio.gather(*(one(request) for request in requests))
    return rows
