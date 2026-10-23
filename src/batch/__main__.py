"""새벽 배치를 손으로 한 번 돌린다. 새벽까지 기다리지 않고 확인할 때 쓴다.

    uv run python -m src.batch plan      # 무엇을 몇 개 만들지만 본다 (생성·전송 없음)
    uv run python -m src.batch generate  # 01:00 생성 배치
    uv run python -m src.batch send      # 03:00 전송 배치

    uv run python -m src.batch generate --only battle  # 한 단계만 (daily|battle|normal)

스케줄러와 같은 함수를 부르므로 advisory lock도 똑같이 잡는다.
서버에서 같은 배치가 도는 중이면 건너뛴다.
"""

import argparse
import asyncio
from collections.abc import Awaitable, Callable

from src.batch.generate import (
    GENERATE_STEPS,
    build_plan,
    run_generate,
)
from src.batch.send import run_send
from src.core.logging import setup_logging
from src.db.session import close_engine


async def show_plan() -> None:
    """생성 배치가 지금 돈다면 무엇을 몇 개 만들지 출력한다. 아무것도 쓰지 않는다."""
    plan = await build_plan()

    source = (
        "받지 못함 (버퍼 재고만 봄)"
        if plan.spring_item_count is None
        else f"{plan.spring_item_count}개 항목"
    )
    print(f"Spring 재고: {source}")

    print(
        f"버퍼 미전송: 일반 {plan.pending_normal}개, "
        f"데일리 {plan.pending_daily}개, 배틀 {plan.pending_battle}개"
    )
    print(f"생성 계획: 조합 {len(plan.items)}개, 문제 {plan.total}개")
    for item in plan.items:
        print(f"  {item.category.value}/{item.difficulty.value}: {item.count}")


JOBS: dict[str, Callable[..., Awaitable[None]]] = {
    "plan": show_plan,
    "generate": run_generate,
    "send": run_send,
}

# 생성과 전송은 단계 이름이 같다(테스트가 확인한다).
STEPS = list(GENERATE_STEPS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.batch", description="새벽 배치를 한 번 실행한다"
    )
    parser.add_argument("job", choices=JOBS)
    parser.add_argument(
        "--only", choices=STEPS, help="이 단계만 돌린다 (generate, send에서만)"
    )
    args = parser.parse_args(argv)
    if args.only and args.job == "plan":
        parser.error("--only는 generate, send에서만 쓸 수 있다")
    return args


async def main(job: str, only: str | None = None) -> None:
    try:
        if only is None:
            await JOBS[job]()
        else:
            await JOBS[job](only=only)
    finally:
        await close_engine()


if __name__ == "__main__":
    args = parse_args()
    setup_logging()
    asyncio.run(main(args.job, args.only))
