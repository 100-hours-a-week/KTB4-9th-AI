"""LangSmith에서 트레이스별 노드 실행시간을 모아 CSV로 만든다.

측정 스크립트는 요청 1건의 전체 시간만 남긴다. 어느 노드에서 시간이 갔는지는
LangSmith에만 있으므로, 여기서 API로 받아 표로 만든다.

실행:
    uv run python loadtest/collect_node_times.py
    uv run python loadtest/collect_node_times.py --hours 24
    uv run python loadtest/collect_node_times.py --hours 24 --out results/node_times.csv
"""

import argparse
import csv
import statistics
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from itertools import islice
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

load_dotenv(".env", override=False)

# 표에 넣을 그래프 노드. 이 목록에 없는 실행(LLM 호출, 파서, 라우터)은 건너뛴다.
NODE_NAMES = [
    "generate_problem",
    "static_validate",
    "check_duplicate",
    "check_contradiction",
    "generate_ref_code",
    "semantic_validate",
    "generate_testcase",
    "generate_solution_code",
    "generate_nl_keyword",
    "collect_parallel",
    "finalize",
    "discard_problem",
]


def duration_s(run) -> float:
    """실행 하나의 소요시간(초). 끝나지 않았으면 0."""
    if run.start_time is None or run.end_time is None:
        return 0.0
    return (run.end_time - run.start_time).total_seconds()


def collect(client: Client, project: str, hours: int, limit: int) -> list[dict]:
    """최근 hours 시간 안의 트레이스를 모아 노드별 시간을 정리한다."""
    since = datetime.now(UTC) - timedelta(hours=hours)

    print(f"LangSmith에서 실행 기록을 받는 중 (프로젝트: {project}, 최근 {hours}시간)")
    runs = list(islice(client.list_runs(project_name=project, start_time=since), limit))
    print(f"실행 {len(runs)}건 수신. 트레이스별로 묶는 중")

    # trace_id로 묶는다. 한 트레이스 = 요청 1건.
    by_trace: dict[str, list] = defaultdict(list)
    for run in runs:
        if run.trace_id:
            by_trace[str(run.trace_id)].append(run)

    rows: list[dict] = []
    for trace_id, members in by_trace.items():
        # 부모가 없는 실행이 트레이스의 뿌리다.
        roots = [r for r in members if r.parent_run_id is None]
        if not roots:
            continue
        root = roots[0]

        outputs = root.outputs or {}
        discarded = bool(outputs.get("is_discarded"))

        row: dict = {
            "trace_id": trace_id,
            "start_time": root.start_time.astimezone().strftime("%m-%d %H:%M:%S"),
            "difficulty": outputs.get("difficulty")
            or outputs.get("requested_difficulty", ""),
            "category": outputs.get("category")
            or outputs.get("requested_category", ""),
            "status": "DISCARDED" if discarded else "SUCCESS",
            "discard_stage": outputs.get("discard_stage") or "",
            "discard_reason": outputs.get("discard_reason") or "",
            "total_s": round(duration_s(root), 2),
        }

        # 노드별 소요시간 합계와 실행 횟수. 재시도가 있으면 횟수가 2 이상이 된다.
        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for run in members:
            if run.name in NODE_NAMES:
                totals[run.name] += duration_s(run)
                counts[run.name] += 1

        for name in NODE_NAMES:
            row[name] = round(totals[name], 2)
        row["ref_code_attempts"] = counts["generate_ref_code"]
        row["semantic_attempts"] = counts["semantic_validate"]

        rows.append(row)

    rows.sort(key=lambda r: r["start_time"])
    return rows


def summarize(rows: list[dict]) -> None:
    """성공한 트레이스만으로 노드별 중앙값과 평균을 출력한다.

    평균은 이상치 한 건에 크게 끌려가므로 중앙값을 함께 본다.
    두 값의 차이가 크면 꼬리가 긴 분포라는 뜻이다.
    """
    success = [r for r in rows if r["status"] == "SUCCESS"]
    if not success:
        print("성공한 트레이스가 없습니다.")
        return

    totals = [r["total_s"] for r in success]
    median_total = statistics.median(totals)

    print()
    print("=" * 78)
    print(f"성공 {len(success)}건 / 전체 {len(rows)}건")
    print(
        f"전체 소요  중앙값 {median_total:.2f}s"
        f"  평균 {statistics.mean(totals):.2f}s"
        f"  최대 {max(totals):.2f}s"
    )
    print()
    print(f"{'노드':<26}{'중앙값':>9}{'평균':>9}{'비중':>8}{'최소':>9}{'최대':>10}")
    print("-" * 78)

    stats = []
    for name in NODE_NAMES:
        values = [r[name] for r in success if r[name] > 0]
        if not values:
            continue
        stats.append((statistics.median(values), name, values))

    for median, name, values in sorted(stats, reverse=True):
        share = median / median_total * 100
        print(
            f"{name:<26}{median:>8.2f}s{statistics.mean(values):>8.2f}s"
            f"{share:>7.1f}%{min(values):>8.2f}s{max(values):>9.2f}s"
        )

    # 중앙값의 3배를 넘는 건은 따로 표시한다. 평균을 왜곡하는 원인이다.
    threshold = median_total * 3
    outliers = [r for r in rows if r["total_s"] > threshold]
    if outliers:
        print()
        print(f"이상치 (전체 소요가 중앙값의 3배 = {threshold:.0f}s 초과)")
        for row in sorted(outliers, key=lambda r: -r["total_s"]):
            longest = max(
                ((row[n], n) for n in NODE_NAMES if row[n] > 0), default=(0, "-")
            )
            print(
                f"  {row['start_time']}  {row['total_s']:7.2f}s  "
                f"{row['status']:<10} 최장 노드: {longest[1]} {longest[0]:.2f}s"
            )
        print("  → 원인이 확인되지 않으면 집계에서 제외하고 별도로 기록한다.")

    retried = [r for r in rows if r["semantic_attempts"] > 1]
    retried_ok = [r for r in retried if r["status"] == "SUCCESS"]
    print()
    print(f"의미 검증을 2회 이상 실행한 트레이스: {len(retried)}건")
    print(f"  그중 최종 성공: {len(retried_ok)}건")
    if retried:
        print(
            "  → 재시도가 문제를 살려낸 비율: "
            f"{len(retried_ok) / len(retried) * 100:.1f}%"
        )
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="LangSmith 노드별 시간 수집")
    parser.add_argument("--project", default="cosmos-rain", help="LangSmith 프로젝트명")
    parser.add_argument("--hours", type=int, default=12, help="최근 몇 시간")
    parser.add_argument("--limit", type=int, default=6000, help="받을 실행 수 상한")
    parser.add_argument(
        "--out", type=Path, default=Path("results/node_times.csv"), help="출력 CSV"
    )
    args = parser.parse_args()

    client = Client()
    rows = collect(client, args.project, args.hours, args.limit)

    if not rows:
        print("트레이스를 찾지 못했습니다. --hours를 늘리거나 프로젝트명을 확인하세요.")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summarize(rows)
    print(f"CSV: {args.out}")


if __name__ == "__main__":
    main()
