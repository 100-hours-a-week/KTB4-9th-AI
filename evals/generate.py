"""평가 1단계: 고정 요청 목록으로 문제를 생성해 파일에 쌓는다.

generate_problem 노드만 실행한다. 로컬 모델이면 Gemini를 부르지 않는다.
생성이 끝나면 Colab 서버를 꺼도 된다. 검증은 2단계(evals.score)에서 한다.

    # Gemini 기준 (운영 모델 그대로)
    uv run python -m evals.generate --label gemini-baseline

    # vLLM에 띄운 모델 (serve_vllm.ipynb 8번 셀에 나온 이름)
    uv run python -m evals.generate --label qwen3-4b-base --model Qwen/Qwen3-4B \\
        --chat-template-kwargs '{"enable_thinking": false}'

    # 앞 4건만 빠르게 확인
    uv run python -m evals.generate --label smoke --model Qwen/Qwen3-4B --limit 4

결과: evals/results/<label>/generations.jsonl (요청 1건 = 1줄), generate_meta.json
같은 label로 다시 실행하면 끝난 요청은 건너뛰고 이어서 돈다.
--retry-errors를 주면 인프라 에러(연결 끊김 등)로 끝난 요청을 다시 돈다.
"""

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import langsmith
from langchain_core.tracers.langchain import wait_for_all_tracers
from pydantic_core import to_jsonable_python

from evals.common import (
    DEFAULT_REQUESTS,
    EVAL_TAG,
    GEMINI,
    LABEL_PATTERN,
    LANGSMITH_PROJECT,
    RESULTS_DIR,
    Row,
    check_meta,
    configure_langsmith,
    file_sha256,
    load_rows,
    run_concurrently,
    select_pending,
)
from evals.requests import EvalRequest, load_requests
from src.client.llm import LocalModel, get_local_client, use_local_model
from src.core.config import get_settings
from src.core.enums import Category, Difficulty
from src.problem.nodes.generate_problem import PROMPT_VERSION, generate_problem
from src.problem.state import GraphState

ROWS_FILE = "generations.jsonl"
META_FILE = "generate_meta.json"

# 이 값이 다르면 같은 label로 이어 돌리지 않는다 (서로 다른 조건의 생성이 섞인다)
IDENTITY = (
    "model",
    "chat_template_kwargs",
    "temperature",
    "requests_sha256",
    "prompt_version",
)


class GenerationRow(Row):
    """요청 한 건의 생성 결과."""

    label: str
    model: str
    difficulty: Difficulty
    category: Category
    status: Literal["ok", "error"]
    error: str | None = None
    seconds: float
    # generate_problem 노드가 그래프에 넘기는 값 그대로 (node_models 제외).
    # 2단계에서 이 값을 그래프에 다시 넣어 검증한다
    output: dict | None = None
    started_at: datetime


def is_model_error(row: GenerationRow) -> bool:
    """
    모델 탓인 실패인가. 형식에 맞는 응답을 못 낸 경우만 모델 탓으로 본다.

    연결 끊김, 타임아웃, 서버 오류는 인프라 문제라 --retry-errors로 다시 돈다.
    """
    return row.status == "error" and (row.error or "").startswith("LLMOutputParseError")


def is_retryable(row: GenerationRow) -> bool:
    return row.status == "error" and not is_model_error(row)


async def generate_one(
    request: EvalRequest, local: LocalModel | None, *, label: str
) -> GenerationRow:
    """
    요청 하나로 generate_problem 노드를 실행한다. 예외를 올리지 않는다.

    Parameters:
        request (EvalRequest): 생성할 요청
        local (LocalModel | None): 로컬 모델. None이면 노드에 적힌 Gemini 모델
        label (str): 실행 이름. 결과와 LangSmith 트레이스에 남는다

    Returns:
        GenerationRow: 생성한 문제 또는 에러
    """
    model = local.model_name if local else GEMINI
    random.seed(request.seed)
    state = GraphState(
        requested_difficulty=request.difficulty,
        requested_category=request.category,
    )
    started_at = datetime.now(UTC)
    start = time.perf_counter()
    output: dict | None = None
    error: str | None = None

    with langsmith.trace(
        name=f"generate:{label}",
        run_type="chain",
        inputs={"request": request.model_dump(mode="json")},
        tags=[EVAL_TAG, label],
        metadata={
            "eval_label": label,
            "model": model,
            "request_id": request.id,
            "difficulty": request.difficulty.value,
            "category": request.category.value,
        },
    ) as run:
        try:
            if local is None:
                result = await generate_problem(state)
            else:
                with use_local_model(local):
                    result = await generate_problem(state)
            # node_models에는 노드에 적힌 Gemini 이름이 남는다. 모델은 row.model로
            output = to_jsonable_python(
                {key: value for key, value in result.items() if key != "node_models"}
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        run.end(outputs={"output": output, "error": error})

    return GenerationRow(
        request_id=request.id,
        label=label,
        model=model,
        difficulty=request.difficulty,
        category=request.category,
        status="error" if error else "ok",
        error=error,
        seconds=round(time.perf_counter() - start, 3),
        output=output,
        started_at=started_at,
    )


def describe(row: GenerationRow) -> str:
    if row.status == "ok":
        title = (row.output or {}).get("problem_title")
        return f"ok {row.seconds:.0f}s {title}"
    kind = "model-error" if is_model_error(row) else "infra-error"
    return f"{kind} {row.seconds:.0f}s {(row.error or '')[:100]}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="평가 1단계: 문제 생성")
    parser.add_argument("--label", required=True, help="실행 이름. 결과 폴더 이름")
    parser.add_argument("--model", help="vLLM에 요청할 모델 이름. 비우면 Gemini")
    parser.add_argument(
        "--chat-template-kwargs",
        type=json.loads,
        help="JSON. 예: '{\"enable_thinking\": false}'",
    )
    parser.add_argument("--temperature", type=float, help="비우면 모델 기본값 (vLLM)")
    parser.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS)
    parser.add_argument("--limit", type=int, help="앞에서부터 이만큼만 돈다")
    parser.add_argument(
        "--concurrency", type=int, default=4, help="동시에 보낼 요청 수"
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="인프라 에러로 끝난 요청을 다시 돈다",
    )
    parser.add_argument(
        "--project", default=LANGSMITH_PROJECT, help="LangSmith 프로젝트"
    )
    parser.add_argument("--verbose", action="store_true", help="노드 로그도 출력")
    args = parser.parse_args(argv)

    if not LABEL_PATTERN.match(args.label):
        parser.error("--label은 영문, 숫자, . _ - 만 쓸 수 있습니다")
    if args.chat_template_kwargs is not None and not args.model:
        parser.error("--chat-template-kwargs는 --model과 같이 씁니다")
    if args.temperature is not None and not args.model:
        parser.error("--temperature는 --model과 같이 씁니다")
    return args


def build_meta(args: argparse.Namespace) -> dict:
    return {
        "label": args.label,
        "model": args.model or GEMINI,
        "chat_template_kwargs": args.chat_template_kwargs,
        "temperature": args.temperature,
        "requests": str(args.requests),
        "requests_sha256": file_sha256(args.requests),
        "prompt_version": PROMPT_VERSION,
        "langsmith_project": args.project,
        "created_at": datetime.now(UTC).isoformat(),
    }


async def preflight(local: LocalModel | None) -> None:
    """생성에 필요한 서버만 확인한다. 문제가 있으면 바로 끝낸다."""
    if local is None:
        if not get_settings().gemini_api_key:
            raise SystemExit("GEMINI_API_KEY가 없습니다")
        return
    try:
        served = [model.id async for model in get_local_client().models.list()]
    except Exception as error:
        raise SystemExit(
            f"vLLM 서버에 연결할 수 없습니다 (VLLM_URL): {error!r}"
        ) from error
    if local.model_name not in served:
        raise SystemExit(f"서버에 {local.model_name!r}이 없습니다. 서빙 중: {served}")


async def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    if not args.verbose:
        # 요청마다 같은 SDK 권고 경고가 찍혀 진행 줄을 가린다
        logging.getLogger("google_genai").setLevel(logging.ERROR)
    configure_langsmith(args.project)

    local = None
    if args.model:
        local = LocalModel(
            model_name=args.model,
            chat_template_kwargs=args.chat_template_kwargs,
            temperature=args.temperature,
        )
    await preflight(local)

    out_dir = RESULTS_DIR / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = build_meta(args)
    check_meta(out_dir / META_FILE, meta, IDENTITY)

    rows_path = out_dir / ROWS_FILE
    requests = load_requests(args.requests)[: args.limit]
    done = load_rows(rows_path, GenerationRow)
    pending = select_pending(
        requests, done, is_retryable if args.retry_errors else None
    )
    print(
        f"label={args.label} model={meta['model']} project={args.project}\n"
        f"요청 {len(requests)}건 중 완료 {len(requests) - len(pending)}건, "
        f"이번에 {len(pending)}건 생성 (동시 {args.concurrency})",
        flush=True,
    )
    if pending:
        try:
            await run_concurrently(
                pending,
                lambda request: generate_one(request, local, label=args.label),
                rows_path=rows_path,
                concurrency=args.concurrency,
                describe=describe,
            )
        finally:
            wait_for_all_tracers()  # 끝나기 전에 LangSmith로 못 보낸 트레이스를 보낸다

    rows = [row for row in load_rows(rows_path, GenerationRow).values()]
    counts = Counter(
        "ok"
        if r.status == "ok"
        else "model-error"
        if is_model_error(r)
        else "infra-error"
        for r in rows
    )
    print(f"끝. {rows_path}: {dict(counts)}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("중단됨. 같은 명령으로 다시 실행하면 이어서 돈다.", file=sys.stderr)
