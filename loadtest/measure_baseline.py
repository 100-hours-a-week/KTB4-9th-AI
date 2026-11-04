"""온디맨드 문제 생성의 기준 성능을 측정한다.

요청을 하나씩 순차로 보내고, 각 요청의 소요시간과 결과를 기록한다.
동시 요청의 영향을 배제한 기준값을 얻는 것이 목적이므로 병렬로 보내지 않는다.

정상 생성과 폐기를 분리해서 집계한다. 폐기된 요청은 그래프 중간에 끝나므로
소요시간이 짧고, 함께 집계하면 분위수가 실제보다 낮게 나온다.

실행:
    uv run python loadtest/measure_baseline.py
    uv run python loadtest/measure_baseline.py --count 20
    uv run python loadtest/measure_baseline.py --out results/baseline_1102.csv
"""

import argparse
import asyncio
import csv
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

import httpx

URL = "http://localhost:8000/api/llm/problem"

# RANDOM은 그래프가 직접 고르므로 제외한다.
CATEGORIES = [
    "ARRAY",
    "STRING",
    "DP",
    "GRAPH",
    "TREE",
    "STACK_QUEUE",
    "BINARY_SEARCH",
    "GREEDY",
    "BACKTRACKING",
    "TWO_POINTER",
    "HASH",
    "HEAP",
    "SORTING",
    "IMPLEMENTATION",
    "BRUTE_FORCE",
    "MATH",
]
DIFFICULTIES = ["LV1", "LV2", "LV3", "LV4", "LV5"]

# 서버에 전체 타임아웃이 없으므로 클라이언트에서 끊는다.
REQUEST_TIMEOUT_S = 300.0

# 폐기 응답의 message에서 단계와 사유를 꺼내는 패턴.
# 예: "문제를 만들지 못했습니다 (check_duplicate: DUPLICATE)"
DISCARD_PATTERN = re.compile(r"\(([^:]+):\s*([^)]+)\)")


def build_requests(count: int) -> list[tuple[str, str]]:
    """난이도와 카테고리가 고르게 섞이도록 요청 목록을 만든다."""
    pairs = [(d, c) for c in CATEGORIES for d in DIFFICULTIES]
    repeated = (pairs * (count // len(pairs) + 1))[:count]
    return repeated


def parse_discard(message: str) -> tuple[str, str]:
    """폐기 메시지에서 단계와 사유를 꺼낸다. 못 찾으면 빈 문자열."""
    match = DISCARD_PATTERN.search(message or "")
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", ""


async def run_one(client: httpx.AsyncClient, difficulty: str, category: str) -> dict:
    """요청 하나를 보내고 결과를 기록한다. 예외도 결과로 남긴다."""
    started = time.monotonic()
    started_at = datetime.now().strftime("%H:%M:%S")

    try:
        response = await client.post(
            URL, json={"difficulty": difficulty, "category": category}
        )
        elapsed = time.monotonic() - started
        body = response.json()
    except Exception as error:
        return {
            "started_at": started_at,
            "difficulty": difficulty,
            "category": category,
            "elapsed_s": round(time.monotonic() - started, 2),
            "status": "ERROR",
            "discard_stage": "",
            "discard_reason": "",
            "detail": repr(error)[:120],
        }

    if response.status_code == 200 and body.get("success"):
        return {
            "started_at": started_at,
            "difficulty": difficulty,
            "category": category,
            "elapsed_s": round(elapsed, 2),
            "status": "SUCCESS",
            "discard_stage": "",
            "discard_reason": "",
            "detail": (body.get("problem") or {}).get("problemTitle", ""),
        }

    stage, reason = parse_discard(body.get("message", ""))
    return {
        "started_at": started_at,
        "difficulty": difficulty,
        "category": category,
        "elapsed_s": round(elapsed, 2),
        "status": "DISCARDED",
        "discard_stage": stage,
        "discard_reason": reason,
        "detail": body.get("errorCode", ""),
    }


def percentile(values: list[float], ratio: float) -> float:
    """정렬된 값에서 분위수를 구한다. 선형 보간 없이 가장 가까운 순위를 쓴다."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(len(ordered) * ratio), len(ordered) - 1)
    return ordered[index]


def summarize(rows: list[dict]) -> None:
    """정상 생성과 폐기를 나눠 집계해 출력한다."""
    success = [r["elapsed_s"] for r in rows if r["status"] == "SUCCESS"]
    discarded = [r for r in rows if r["status"] == "DISCARDED"]
    errors = [r for r in rows if r["status"] == "ERROR"]

    print()
    print("=" * 62)
    print(
        f"전체 {len(rows)}건  |  성공 {len(success)}  폐기 {len(discarded)}  오류 {len(errors)}"
    )
    if rows:
        print(f"유효 문제 생성률: {len(success) / len(rows) * 100:.1f}%")

    if success:
        print()
        print("--- 정상 생성만 (폐기 제외) ---")
        print(f"  건수    {len(success)}")
        print(f"  최소    {min(success):.2f}s")
        print(f"  p50     {percentile(success, 0.50):.2f}s")
        print(f"  평균    {statistics.mean(success):.2f}s")
        print(f"  p90     {percentile(success, 0.90):.2f}s")
        print(f"  p95     {percentile(success, 0.95):.2f}s")
        print(f"  최대    {max(success):.2f}s")
        over = [v for v in success if v > 30]
        print(f"  30초 초과  {len(over)}건 ({len(over) / len(success) * 100:.1f}%)")

    if discarded:
        print()
        print("--- 폐기 (별도 집계) ---")
        times = [r["elapsed_s"] for r in discarded]
        print(f"  건수    {len(discarded)}")
        print(f"  p50     {percentile(times, 0.50):.2f}s")
        print(f"  평균    {statistics.mean(times):.2f}s")
        print()
        print("  폐기 단계별:")
        stages: dict[str, list[float]] = {}
        for row in discarded:
            key = f"{row['discard_stage']} / {row['discard_reason']}"
            stages.setdefault(key, []).append(row["elapsed_s"])
        for key, times in sorted(stages.items(), key=lambda x: -len(x[1])):
            print(f"    {key:45} {len(times):3}건  평균 {statistics.mean(times):6.2f}s")

    if errors:
        print()
        print("--- 오류 ---")
        for row in errors:
            print(f"  {row['difficulty']}/{row['category']}: {row['detail']}")

    print("=" * 62)


async def main() -> None:
    parser = argparse.ArgumentParser(description="온디맨드 생성 기준 성능 측정")
    parser.add_argument("--count", type=int, default=72, help="요청 건수")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(f"baseline_{datetime.now():%m%d_%H%M}.csv"),
        help="결과 CSV 경로",
    )
    args = parser.parse_args()

    requests = build_requests(args.count)
    print(f"총 {len(requests)}건을 순차로 보냅니다. 중간에 끊으려면 Control + C.")
    print()

    rows: list[dict] = []
    overall_started = time.monotonic()

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
        for index, (difficulty, category) in enumerate(requests, start=1):
            row = await run_one(client, difficulty, category)
            rows.append(row)

            mark = {"SUCCESS": "O", "DISCARDED": "X", "ERROR": "!"}[row["status"]]
            note = (
                f"{row['discard_stage']}/{row['discard_reason']}"
                if row["status"] == "DISCARDED"
                else row["detail"][:28]
            )
            print(
                f"[{index:3}/{len(requests)}] {mark} "
                f"{row['difficulty']}/{row['category']:16} "
                f"{row['elapsed_s']:7.2f}s  {note}"
            )

            # 결과를 매 건 저장한다. 중간에 끊겨도 기록이 남는다.
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    print()
    print(f"전체 소요 {time.monotonic() - overall_started:.0f}초")
    summarize(rows)
    print(f"CSV: {args.out}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n중단했습니다. 그때까지의 결과는 CSV에 남아 있습니다.")
