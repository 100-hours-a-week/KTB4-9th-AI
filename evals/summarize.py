"""평가 3단계: 생성·채점 결과를 모델별 비교표로 만든다.

    uv run python -m evals.summarize gemini-baseline qwen3-4b-base qwen3-4b-ft
    uv run python -m evals.summarize gemini-baseline qwen3-4b-base \\
        --diversity --out summary.md

첫 번째 label을 기준으로 요청별 짝 비교도 만든다.
표는 마크다운이라 PR이나 Notion에 그대로 붙여 넣을 수 있다.

요청 하나의 결과는 다음 중 하나다.
- 통과: 의미 검증까지 통과
- 폐기: 생성 형식 실패, 정적 검증 폐기, 의미 검증 폐기 (모델 탓)
- 미확정: 아직 생성·채점 안 함, 또는 인프라 장애로 끝남 → 비율 계산에서 뺀다
"""

import argparse
import asyncio
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from evals.common import RESULTS_DIR, load_rows
from evals.generate import META_FILE as GENERATE_META_FILE
from evals.generate import ROWS_FILE as GENERATIONS_FILE
from evals.generate import GenerationRow, is_model_error
from evals.requests import EvalRequest, load_requests
from evals.score import ROWS_FILE as SCORES_FILE
from evals.score import ScoreRow, generation_sha, is_infra_failure
from src.core.enums import Category, Difficulty

PASSED = "PASSED"
FORMAT_FAILED = "generate_problem:FORMAT_FAILED"  # 형식에 맞는 응답을 못 냄
NOT_GENERATED = "NOT_GENERATED"
NOT_SCORED = "NOT_SCORED"
INFRA = "INFRA"
UNDETERMINED = {NOT_GENERATED, NOT_SCORED, INFRA}

DUPLICATE_SIMILARITY = 0.87  # check_duplicate와 같은 기준
EMBEDDINGS_FILE = "embeddings.json"


@dataclass
class Outcome:
    request: EvalRequest
    result: str  # PASSED, "단계:사유", 또는 미확정 표시
    generation: GenerationRow | None = None
    score: ScoreRow | None = None

    @property
    def determined(self) -> bool:
        return self.result not in UNDETERMINED

    @property
    def passed(self) -> bool:
        return self.result == PASSED


@dataclass
class LabelResult:
    label: str
    model: str
    outcomes: list[Outcome]
    duplicates: int | None = None  # --diversity일 때만
    duplicate_pool: int | None = None
    stats: dict = field(default_factory=dict)

    @property
    def determined(self) -> list[Outcome]:
        return [o for o in self.outcomes if o.determined]


def classify(
    request: EvalRequest, generation: GenerationRow | None, score: ScoreRow | None
) -> str:
    """요청 하나의 최종 결과를 정한다."""
    if generation is None:
        return NOT_GENERATED
    if generation.status != "ok":
        return FORMAT_FAILED if is_model_error(generation) else INFRA
    if score is None or score.generation_sha != generation_sha(generation.output):
        return NOT_SCORED
    if is_infra_failure(score):
        return INFRA
    if score.status == "passed":
        return PASSED
    return f"{score.discard_stage}:{score.discard_reason}"


def collect(label: str, results_dir: Path = RESULTS_DIR) -> LabelResult:
    """
    label 하나의 생성·채점 결과를 요청 목록 순서대로 모은다.

    Raises:
        SystemExit: 1단계 결과가 없는 경우
    """
    out_dir = results_dir / label
    meta_path = out_dir / GENERATE_META_FILE
    if not meta_path.exists():
        raise SystemExit(f"{out_dir}에 결과가 없습니다")
    meta = json.loads(meta_path.read_text())
    requests = load_requests(Path(meta["requests"]))
    generations = load_rows(out_dir / GENERATIONS_FILE, GenerationRow)
    scores = load_rows(out_dir / SCORES_FILE, ScoreRow)
    outcomes = [
        Outcome(
            request=request,
            result=classify(
                request, generations.get(request.id), scores.get(request.id)
            ),
            generation=generations.get(request.id),
            score=scores.get(request.id),
        )
        for request in requests
    ]
    return LabelResult(label=label, model=meta["model"], outcomes=outcomes)


# ── 통계 ──────────────────────────────────────────────────────────────


def wilson_interval(passed: int, total: int) -> tuple[float, float]:
    """통과율의 95% 신뢰구간. 표본이 작을 때도 0~1을 벗어나지 않는다."""
    if total == 0:
        return 0.0, 0.0
    z = 1.96
    p = passed / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    half /= 1 + z * z / total
    return max(0.0, center - half), min(1.0, center + half)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1]


def pct(part: int, total: int) -> str:
    return f"{part / total:.0%}" if total else "-"


def seconds(value: float | None) -> str:
    return f"{value:.1f}s" if value is not None else "-"


# ── 표 ────────────────────────────────────────────────────────────────


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def overview_table(results: list[LabelResult]) -> str:
    headers = [
        "label",
        "모델",
        "확정/전체",
        "**통과율** (95% 구간)",
        "생성 형식 실패",
        "정적 검증 폐기",
        "의미 검증 폐기",
        "생성 시간 p50 / p90",
        "의미 검증 시도 평균",
    ]
    rows = []
    for r in results:
        determined = r.determined
        total = len(determined)
        passed = sum(o.passed for o in determined)
        low, high = wilson_interval(passed, total)
        by_stage = Counter(o.result.split(":")[0] for o in determined)
        gen_seconds = [
            o.generation.seconds
            for o in r.outcomes
            if o.generation and o.generation.status == "ok"
        ]
        attempts = [
            o.score.node_calls.get("generate_ref_code", 0)
            for o in determined
            if o.score and o.score.node_calls.get("generate_ref_code")
        ]
        rows.append(
            [
                r.label,
                r.model,
                f"{total}/{len(r.outcomes)}",
                f"**{pct(passed, total)}** ({low:.0%}~{high:.0%})" if total else "-",
                pct(by_stage["generate_problem"], total),
                pct(by_stage["static_validate"], total),
                pct(by_stage["semantic_validate"], total),
                f"{seconds(percentile(gen_seconds, 50))} / "
                f"{seconds(percentile(gen_seconds, 90))}",
                f"{statistics.mean(attempts):.2f}" if attempts else "-",
            ]
        )
    return markdown_table(headers, rows)


def reason_table(results: list[LabelResult]) -> str:
    """폐기 사유별 건수. 각 label의 확정 건수 대비 비율도 붙인다."""
    reasons = sorted({o.result for r in results for o in r.determined if not o.passed})
    if not reasons:
        return "폐기된 문제 없음"
    rows = []
    for reason in reasons:
        row = [reason.replace(":", " / ")]
        for r in results:
            count = sum(o.result == reason for o in r.determined)
            row.append(f"{count} ({pct(count, len(r.determined))})" if count else "-")
        rows.append(row)
    return markdown_table(["폐기 단계 / 사유", *(r.label for r in results)], rows)


def breakdown_table(results: list[LabelResult], key: str) -> str:
    """난이도별 또는 카테고리별 통과율 (통과/확정)."""
    values = sorted(
        {getattr(o.request, key) for r in results for o in r.outcomes},
        key=lambda v: list(Difficulty if key == "difficulty" else Category).index(v),
    )
    rows = []
    for value in values:
        row = [value.value]
        for r in results:
            group = [o for o in r.determined if getattr(o.request, key) == value]
            passed = sum(o.passed for o in group)
            row.append(f"{pct(passed, len(group))} ({passed}/{len(group)})")
        rows.append(row)
    title = "난이도" if key == "difficulty" else "카테고리"
    return markdown_table([title, *(r.label for r in results)], rows)


def paired_table(results: list[LabelResult]) -> str:
    """첫 label과 같은 요청끼리 비교한다. 둘 다 확정된 요청만 센다."""
    base = results[0]
    base_by_id = {o.request.id: o for o in base.determined}
    rows = []
    for r in results[1:]:
        pairs = [
            (base_by_id[o.request.id], o)
            for o in r.determined
            if o.request.id in base_by_id
        ]
        both = sum(b.passed and o.passed for b, o in pairs)
        only_base = sum(b.passed and not o.passed for b, o in pairs)
        only_this = sum(o.passed and not b.passed for b, o in pairs)
        neither = len(pairs) - both - only_base - only_this
        rows.append(
            [
                r.label,
                str(len(pairs)),
                str(both),
                str(only_base),
                str(only_this),
                str(neither),
            ]
        )
    return markdown_table(
        [
            f"label (기준: {base.label})",
            "짝 수",
            "둘 다 통과",
            "기준만 통과",
            "이 모델만 통과",
            "둘 다 실패",
        ],
        rows,
    )


def diversity_table(results: list[LabelResult]) -> str:
    rows = [
        [
            r.label,
            str(r.duplicate_pool),
            f"{r.duplicates} ({pct(r.duplicates or 0, r.duplicate_pool or 0)})",
        ]
        for r in results
        if r.duplicates is not None
    ]
    return markdown_table(
        ["label", "생성된 문제", f"앞 문제와 중복 (유사도 ≥ {DUPLICATE_SIMILARITY})"],
        rows,
    )


# ── 다양성 ────────────────────────────────────────────────────────────


async def count_duplicates(
    result: LabelResult,
    embed: Callable[[str], Awaitable[list[float]]],
    cache_path: Path | None = None,
) -> None:
    """
    같은 카테고리의 앞선 문제와 algorithm_core가 비슷하면 중복으로 센다.

    운영 중복 검사와 같은 기준이다. 생성에 성공한 문제 전부를 요청 목록 순서로 보고,
    운영 DB의 문제와는 비교하지 않는다.
    임베딩은 생성 결과별로 캐시해 다시 계산하지 않는다.
    """
    cache: dict[str, list[float]] = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text())

    seen: dict[Category, list[list[float]]] = defaultdict(list)
    duplicates = pool = 0
    for outcome in result.outcomes:
        generation = outcome.generation
        if generation is None or generation.status != "ok":
            continue
        core = (generation.output or {}).get("algorithm_core")
        if not core:
            continue
        key = generation_sha(generation.output)
        if key not in cache:
            cache[key] = await embed(core)
        vector = cache[key]
        category = outcome.request.category
        if any(
            sum(a * b for a, b in zip(vector, other, strict=True))
            >= DUPLICATE_SIMILARITY
            for other in seen[category]
        ):
            duplicates += 1
        seen[category].append(vector)
        pool += 1

    if cache_path:
        cache_path.write_text(json.dumps(cache))
    result.duplicates, result.duplicate_pool = duplicates, pool


# ── 실행 ──────────────────────────────────────────────────────────────


def render(results: list[LabelResult]) -> str:
    sections = [
        "## 요약",
        overview_table(results),
        "- 비율의 분모는 확정된 요청이다. 미생성·미채점·인프라 장애는 뺀다.\n"
        "- 의미 검증 시도 = 레퍼런스 코드를 만든 횟수 (1이면 한 번에 판정).",
        "## 폐기 사유",
        reason_table(results),
        "## 난이도별 통과율",
        breakdown_table(results, "difficulty"),
        "## 카테고리별 통과율",
        breakdown_table(results, "category"),
    ]
    if len(results) > 1:
        sections += ["## 같은 요청끼리 비교", paired_table(results)]
    if any(r.duplicates is not None for r in results):
        sections += ["## 다양성", diversity_table(results)]
    return "\n\n".join(sections) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="평가 3단계: 모델별 비교표")
    parser.add_argument("labels", nargs="+", help="비교할 label. 첫 번째가 기준")
    parser.add_argument(
        "--diversity",
        action="store_true",
        help="algorithm_core 임베딩으로 중복을 센다 (임베딩 API 호출)",
    )
    parser.add_argument("--out", type=Path, help="마크다운으로 저장할 파일")
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results = [collect(label) for label in args.labels]
    if args.diversity:
        from src.client.embedding import embed_text

        for result in results:
            await count_duplicates(
                result, embed_text, RESULTS_DIR / result.label / EMBEDDINGS_FILE
            )
    text = render(results)
    print(text)
    if args.out:
        args.out.write_text(text)
        print(f"저장: {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
