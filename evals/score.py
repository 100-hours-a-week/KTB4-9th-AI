"""평가 2단계: 생성된 문제를 운영 검증에 통과시켜 채점한다.

운영 그래프(build_graph)를 그대로 쓰고 다음만 바꾼다.
- generate_problem: 1단계에서 저장한 문제를 그대로 돌려준다 (모델을 다시 부르지 않는다)
- check_duplicate: 끈다. 로컬 DB에 쌓인 문제가 사람·시점마다 달라 공정한 비교가 안 된다
- discard_problem, finalize: DB에 쓰지 않는다
- 병렬 생성 노드(테스트케이스, 정답 코드, 키워드) 직전에서 멈춘다

그래서 정적 검증 → 레퍼런스 코드 → judge0 실행 → 의미 검증(재시도 포함)까지
운영과 같은 기준으로 돈다. Colab 서버는 필요 없고 Gemini와 judge0만 쓴다.

    uv run python -m evals.score --label qwen3-4b-base

입력: evals/results/<label>/generations.jsonl (1단계 결과)
결과: evals/results/<label>/scores.jsonl (요청 1건 = 1줄), score_meta.json
같은 label로 다시 실행하면 끝난 요청은 건너뛴다.
1단계에서 다시 생성한 문제는 다시 채점한다.
--retry-errors를 주면 인프라 문제(Gemini·judge0 장애)로 끝난 요청을 다시 돈다.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx
from langchain_core.tracers.langchain import wait_for_all_tracers
from langgraph.graph.state import CompiledStateGraph

from evals.common import (
    EVAL_TAG,
    LABEL_PATTERN,
    LANGSMITH_PROJECT,
    RESULTS_DIR,
    Row,
    check_meta,
    configure_langsmith,
    load_rows,
    run_concurrently,
)
from evals.generate import META_FILE as GENERATE_META_FILE
from evals.generate import ROWS_FILE as GENERATIONS_FILE
from evals.generate import GenerationRow
from evals.requests import EvalRequest, load_requests
from src.core.config import get_settings
from src.core.enums import Category, Difficulty, DiscardReason
from src.problem.graph import DEFAULT_NODES, PARALLEL_NODES, NodeFn, build_graph
from src.problem.nodes import generate_ref_code, semantic_validate
from src.problem.state import GraphState

ROWS_FILE = "scores.jsonl"
META_FILE = "score_meta.json"

# 문제 탓이 아니라 채점 쪽(Gemini·judge0) 장애로 폐기된 사유. 다시 돌린다
INFRA_DISCARDS = {DiscardReason.LLM_ERROR, DiscardReason.EXECUTOR_ERROR}

# 채점 기준. 이 값이 다르면 같은 label로 이어 채점하지 않는다
IDENTITY = ("ref_code_model", "ref_code_prompt", "semantic_model", "semantic_prompt")


class ScoreRow(Row):
    """요청 한 건의 채점 결과."""

    label: str
    model: str  # 문제를 생성한 모델
    difficulty: Difficulty
    category: Category
    generation_sha: str  # 채점한 생성 결과. 다시 생성하면 달라져 다시 채점한다
    status: Literal["passed", "discarded", "error"]
    discard_stage: str | None = None
    discard_reason: str | None = None
    discard_detail: str | None = None
    error_node: str | None = None
    error: str | None = None
    seconds: float
    node_seconds: dict[str, float]
    node_calls: dict[str, int]  # generate_ref_code 횟수 = 의미 검증 시도 횟수
    started_at: datetime


@dataclass
class ScoreTrace:
    """그래프 한 번 실행 동안 노드들이 채우는 기록."""

    output: dict  # 재생할 생성 결과
    seconds: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_node: str | None = None


# 요청마다 따로 둔다. 동시에 도는 요청끼리 섞이지 않는다
_trace: ContextVar[ScoreTrace] = ContextVar("score_trace")


def generation_sha(output: dict) -> str:
    text = json.dumps(output, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


async def replay_generation(state: GraphState) -> dict:
    """generate_problem 대신. 1단계에서 저장한 문제를 그대로 돌려준다."""
    return dict(_trace.get().output)


async def skip(state: GraphState) -> dict:
    """중복 검사, DB 저장, 폐기 로그 대신. 아무것도 하지 않는다."""
    return {}


def timed(name: str, node: NodeFn) -> NodeFn:
    """노드 실행 시간과 횟수를 기록한다. 예외가 나면 어느 노드인지 남긴다."""

    async def wrapper(state: GraphState) -> dict:
        trace = _trace.get()
        trace.calls[name] += 1
        start = time.perf_counter()
        try:
            return await node(state)
        except Exception:
            if trace.error_node is None:
                trace.error_node = name
            raise
        finally:
            trace.seconds[name] += time.perf_counter() - start

    return wrapper


def build_score_graph(nodes: dict[str, NodeFn] | None = None) -> CompiledStateGraph:
    """
    채점용 그래프를 조립한다. 병렬 생성 노드 직전에서 멈추는 것은 실행할 때 정한다.

    Parameters:
        nodes (dict[str, NodeFn] | None): 기본 노드. 테스트에서 가짜 노드를 넣는다

    Returns:
        CompiledStateGraph: 실행 가능한 그래프
    """
    nodes = {
        **(nodes or DEFAULT_NODES),
        "generate_problem": replay_generation,
        "check_duplicate": skip,
        "discard_problem": skip,
        "finalize": skip,
    }
    return build_graph({name: timed(name, node) for name, node in nodes.items()})


async def score_one(
    graph: CompiledStateGraph,
    request: EvalRequest,
    generation: GenerationRow,
    *,
    label: str,
) -> ScoreRow:
    """
    생성된 문제 하나를 채점한다. 예외를 올리지 않는다.

    Parameters:
        graph (CompiledStateGraph): build_score_graph로 만든 그래프
        request (EvalRequest): 생성할 때 쓴 요청
        generation (GenerationRow): 1단계 결과 (status가 ok인 것)
        label (str): 실행 이름

    Returns:
        ScoreRow: 통과, 폐기, 에러 중 하나
    """
    assert generation.output is not None
    trace = ScoreTrace(output=generation.output)
    token = _trace.set(trace)
    state = GraphState(
        requested_difficulty=request.difficulty,
        requested_category=request.category,
    )
    config = {
        "run_name": f"score:{label}",
        "tags": [EVAL_TAG, label],
        "metadata": {
            "eval_label": label,
            "model": generation.model,
            "request_id": request.id,
            "difficulty": request.difficulty.value,
            "category": request.category.value,
        },
    }
    started_at = datetime.now(UTC)
    start = time.perf_counter()
    result: GraphState | None = None
    error: Exception | None = None
    try:
        values = await graph.ainvoke(
            state, config=config, interrupt_before=PARALLEL_NODES
        )
        result = GraphState(**values)
    except Exception as exc:
        error = exc
    finally:
        _trace.reset(token)

    row = {
        "request_id": request.id,
        "label": label,
        "model": generation.model,
        "difficulty": request.difficulty,
        "category": request.category,
        "generation_sha": generation_sha(generation.output),
        "seconds": round(time.perf_counter() - start, 3),
        "node_seconds": {k: round(v, 3) for k, v in trace.seconds.items()},
        "node_calls": dict(trace.calls),
        "started_at": started_at,
    }
    if result is None:
        return ScoreRow(
            **row,
            status="error",
            error_node=trace.error_node,
            error=f"{type(error).__name__}: {error}",
        )
    if result.is_discarded:
        return ScoreRow(
            **row,
            status="discarded",
            discard_stage=result.discard_stage,
            discard_reason=result.discard_reason,
            discard_detail=result.discard_detail,
        )
    return ScoreRow(**row, status="passed")


def is_infra_failure(row: ScoreRow) -> bool:
    """문제 탓이 아니라 채점 쪽 장애로 끝났는가. 다시 돌리고, 비교에서 뺀다."""
    if row.status == "error":
        return True
    return row.status == "discarded" and row.discard_reason in INFRA_DISCARDS


def select_scoring(
    requests: list[EvalRequest],
    generations: dict[str, GenerationRow],
    scores: dict[str, ScoreRow],
    *,
    retry_errors: bool,
) -> list[EvalRequest]:
    """
    채점할 요청을 고른다.

    생성에 성공한 요청만 대상이다. 채점한 적이 없거나, 채점 뒤에 다시 생성됐거나,
    retry_errors이고 인프라 장애로 끝난 요청을 고른다.
    """
    pending = []
    for request in requests:
        generation = generations.get(request.id)
        if generation is None or generation.status != "ok":
            continue
        score = scores.get(request.id)
        if score is None or score.generation_sha != generation_sha(generation.output):
            pending.append(request)
        elif retry_errors and is_infra_failure(score):
            pending.append(request)
    return pending


def describe(row: ScoreRow) -> str:
    attempts = row.node_calls.get("generate_ref_code", 0)
    tail = f"{row.seconds:.0f}s ref×{attempts}"
    if row.status == "passed":
        return f"passed {tail}"
    if is_infra_failure(row):
        detail = row.discard_reason or f"{row.error_node}: {row.error}"
        return f"infra {tail} {detail[:100]}"
    return f"discarded {row.discard_stage}/{row.discard_reason} {tail}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="평가 2단계: 생성된 문제 채점")
    parser.add_argument("--label", required=True, help="1단계에서 쓴 실행 이름")
    parser.add_argument("--limit", type=int, help="요청 목록 앞에서부터 이만큼만")
    parser.add_argument(
        "--concurrency", type=int, default=4, help="동시에 채점할 문제 수"
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="인프라 장애(Gemini·judge0)로 끝난 요청을 다시 돈다",
    )
    parser.add_argument(
        "--project", default=LANGSMITH_PROJECT, help="LangSmith 프로젝트"
    )
    parser.add_argument("--verbose", action="store_true", help="노드 로그도 출력")
    args = parser.parse_args(argv)
    if not LABEL_PATTERN.match(args.label):
        parser.error("--label은 영문, 숫자, . _ - 만 쓸 수 있습니다")
    return args


def build_meta(args: argparse.Namespace, model: str) -> dict:
    return {
        "label": args.label,
        "model": model,
        "ref_code_model": generate_ref_code.MODEL_NAME,
        "ref_code_prompt": generate_ref_code.PROMPT_VERSION,
        "semantic_model": semantic_validate.MODEL_NAME,
        "semantic_prompt": semantic_validate.PROMPT_VERSION,
        "langsmith_project": args.project,
        "created_at": datetime.now(UTC).isoformat(),
    }


async def preflight() -> None:
    """채점에 필요한 Gemini 키와 judge0를 확인한다."""
    settings = get_settings()
    if not settings.gemini_api_key:
        raise SystemExit("GEMINI_API_KEY가 없습니다 (검증 노드가 Gemini를 씁니다)")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            (await client.get(f"{settings.judge0_url}/about")).raise_for_status()
    except httpx.HTTPError as error:
        raise SystemExit(
            f"judge0에 연결할 수 없습니다 ({settings.judge0_url}): {error!r}"
        ) from error


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

    out_dir = RESULTS_DIR / args.label
    generate_meta_path = out_dir / GENERATE_META_FILE
    if not generate_meta_path.exists():
        raise SystemExit(
            f"{out_dir}에 1단계 결과가 없습니다. evals.generate부터 돌리세요."
        )
    generate_meta = json.loads(generate_meta_path.read_text())
    await preflight()

    check_meta(out_dir / META_FILE, build_meta(args, generate_meta["model"]), IDENTITY)

    rows_path = out_dir / ROWS_FILE
    requests = load_requests(Path(generate_meta["requests"]))[: args.limit]
    generations = load_rows(out_dir / GENERATIONS_FILE, GenerationRow)
    scores = load_rows(rows_path, ScoreRow)
    pending = select_scoring(
        requests, generations, scores, retry_errors=args.retry_errors
    )
    failed = sum(
        1 for r in requests if (g := generations.get(r.id)) and g.status != "ok"
    )
    missing = sum(1 for r in requests if r.id not in generations)
    print(
        f"label={args.label} model={generate_meta['model']} project={args.project}\n"
        f"요청 {len(requests)}건: 생성 실패 {failed}건, 미생성 {missing}건은 "
        "채점 안 함, "
        f"이번에 {len(pending)}건 채점 (동시 {args.concurrency})",
        flush=True,
    )
    if pending:
        graph = build_score_graph()
        try:
            await run_concurrently(
                pending,
                lambda request: score_one(
                    graph, request, generations[request.id], label=args.label
                ),
                rows_path=rows_path,
                concurrency=args.concurrency,
                describe=describe,
            )
        finally:
            wait_for_all_tracers()  # 끝나기 전에 LangSmith로 못 보낸 트레이스를 보낸다

    final = load_rows(rows_path, ScoreRow)
    counts = Counter(
        "infra" if is_infra_failure(r) else r.status
        for r in final.values()
        if r.request_id in {q.id for q in requests}
    )
    print(f"끝. {rows_path}: {dict(counts)}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("중단됨. 같은 명령으로 다시 실행하면 이어서 돈다.", file=sys.stderr)
