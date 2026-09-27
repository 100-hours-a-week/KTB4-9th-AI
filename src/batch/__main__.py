"""새벽 배치를 손으로 한 번 돌린다. 새벽까지 기다리지 않고 확인할 때 쓴다.

    uv run python -m src.batch plan      # 무엇을 몇 개 만들지만 본다 (생성·전송 없음)
    uv run python -m src.batch generate  # 01:00 생성 배치
    uv run python -m src.batch send      # 03:00 전송 배치

스케줄러와 같은 함수를 부르므로 advisory lock도 똑같이 잡는다.
서버에서 같은 배치가 도는 중이면 건너뛴다.
"""

import argparse
import asyncio
from collections.abc import Awaitable, Callable

from src.batch.generate import (
    fetch_demands,
    interleave,
    load_pending,
    plan_generation,
    run_generate,
)
from src.batch.send import run_send
from src.core.enums import ProblemPurpose
from src.core.logging import setup_logging
from src.db.session import close_engine


async def show_plan() -> None:
    """생성 배치가 지금 돈다면 무엇을 몇 개 만들지 출력한다. 아무것도 쓰지 않는다."""
    demands = await fetch_demands()
    pending = await load_pending(ProblemPurpose.NORMAL)
    daily_pending = sum((await load_pending(ProblemPurpose.DAILY)).values())
    plan = plan_generation(demands or [], pending)

    source = (
        "받지 못함 (버퍼 재고만 봄)" if demands is None else f"{len(demands)}개 항목"
    )
    print(f"Spring 재고: {source}")
    print(f"버퍼 미전송: 일반 {sum(pending.values())}개, 데일리 {daily_pending}개")
    print(f"생성 계획: 조합 {len(plan)}개, 문제 {len(interleave(plan))}개")
    for (category, difficulty), need in plan.items():
        print(f"  {category.value}/{difficulty.value}: {need}")


JOBS: dict[str, Callable[[], Awaitable[None]]] = {
    "plan": show_plan,
    "generate": run_generate,
    "send": run_send,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.batch", description="새벽 배치를 한 번 실행한다"
    )
    parser.add_argument("job", choices=JOBS)
    return parser.parse_args(argv)


async def main(job: str) -> None:
    try:
        await JOBS[job]()
    finally:
        await close_engine()


if __name__ == "__main__":
    args = parse_args()
    setup_logging()
    asyncio.run(main(args.job))
