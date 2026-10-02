"""새벽 문제 생성 배치.

01:00에 돈다. 데일리 5개, 배틀 1개를 만들고, 조합(카테고리, 난이도)마다
아직 아무도 풀지 않은 문제가 TARGET_STOCK개가 되도록 부족한 만큼 만든다.
만든 문제는 finalize가 버퍼에 쌓고, 03:00 전송 배치가 Spring으로 보낸다.
"""

import asyncio
import logging
import time
from collections import Counter
from collections.abc import Awaitable, Callable

from src.client.backend import get_problem_demands
from src.core.config import get_settings
from src.core.enums import Category, Difficulty, LockKey, ProblemPurpose, Trigger
from src.db.lock import try_advisory_lock
from src.db.repository import BattleProblemRepository, GeneratedProblemRepository
from src.db.session import session_scope
from src.problem.battle.runner import run_battle_graph
from src.problem.daily import DAILY_COUNT, generate_daily_problems
from src.problem.runner import run_problem_graph
from src.schema.batch import ProblemDemand

logger = logging.getLogger(__name__)

TARGET_STOCK = 3  # 조합마다 유지할 미풀이 문제 수
MAX_ATTEMPTS = 3  # 슬롯 하나에 그래프를 돌리는 최대 횟수

# Spring 재고 조회 재시도 간격(초). 첫 시도까지 합쳐 총 3번 시도한다.
DEMAND_RETRY_DELAYS_S = (30, 60)

type ProblemKey = tuple[Category, Difficulty]
type Step = Callable[[], Awaitable[None]]

# Spring은 재고가 0인 조합을 응답에서 뺄 수 있다.
# 그래서 응답이 아니라 이 목록을 기준으로 채운다.
# 난이도를 바깥에 두어 이웃한 조합끼리 카테고리가 다르게 한다(interleave 참고).
ALL_KEYS: list[ProblemKey] = [
    (category, difficulty)
    for difficulty in Difficulty
    for category in Category
    if category != Category.RANDOM
]


def _spring_stock(demands: list[ProblemDemand]) -> dict[ProblemKey, int]:
    """Spring 응답을 조합별 개수로 모은다. 같은 조합이 여러 번 오면 합친다."""
    stock: dict[ProblemKey, int] = {}
    for demand in demands:
        key = (demand.category, demand.difficulty)
        if key in stock:
            logger.warning(
                "Spring 재고에 같은 조합이 중복됨: %s/%s (합산)",
                demand.category.value,
                demand.difficulty.value,
            )
        stock[key] = stock.get(key, 0) + demand.count
    return stock


def plan_generation(
    demands: list[ProblemDemand], pending: dict[ProblemKey, int]
) -> dict[ProblemKey, int]:
    """
    조합마다 몇 개를 만들지 정한다.

    부족분 = TARGET_STOCK - (Spring 재고 + 아직 안 보낸 버퍼 재고).
    Spring 응답에 없는 조합은 재고 0으로 본다. Spring 조회에 실패한 날은
    빈 목록을 넘기면 버퍼 재고만 보고 채운다.

    Parameters:
        demands (list[ProblemDemand]): Spring이 알려 준 조합별 미풀이 문제 수
        pending (dict[ProblemKey, int]): 배치로 만들고 아직 보내지 않은 문제 수

    Returns:
        dict[ProblemKey, int]: 만들 개수. 0개인 조합은 빠진다
    """
    stock = _spring_stock(demands)
    plan: dict[ProblemKey, int] = {}
    for key in ALL_KEYS:
        need = TARGET_STOCK - stock.get(key, 0) - pending.get(key, 0)
        if need > 0:
            plan[key] = need
    return plan


def interleave(plan: dict[ProblemKey, int]) -> list[ProblemKey]:
    """
    만들 문제를 한 줄로 세운다. 같은 카테고리가 연달아 오지 않게 돌아가며 뽑는다.

    중복 검사는 같은 카테고리 안에서 한다. 같은 카테고리를 동시에 만들면
    서로의 임베딩이 아직 DB에 없어 둘 다 중복 검사를 통과할 수 있다.
    모든 조합의 1번째, 모든 조합의 2번째… 순서로 세우고 조합은 ALL_KEYS
    순서(카테고리가 안쪽)를 따르므로, 같은 카테고리 사이에 다른 카테고리들이 낀다.

    Parameters:
        plan (dict[ProblemKey, int]): 조합별 만들 개수

    Returns:
        list[ProblemKey]: 슬롯 목록. 길이는 plan의 합
    """
    rounds = max(plan.values(), default=0)
    return [key for i in range(rounds) for key, need in plan.items() if need > i]


def _label(key: ProblemKey) -> str:
    category, difficulty = key
    return f"{category.value}/{difficulty.value}"


async def fill_slot(key: ProblemKey, semaphore: asyncio.Semaphore) -> bool:
    """
    슬롯 하나를 채운다. 폐기되면 MAX_ATTEMPTS번까지 다시 만든다.

    시도할 때마다 semaphore를 새로 잡는다. 실패한 슬롯은 대기열 맨 뒤로
    가므로, 재시도도 다른 조합들 사이에 흩어져 돈다.

    Parameters:
        key (ProblemKey): 만들 조합
        semaphore (asyncio.Semaphore): 동시에 도는 그래프 수 제한

    Returns:
        bool: 채웠으면 True
    """
    category, difficulty = key
    for _ in range(MAX_ATTEMPTS):
        async with semaphore:
            problem = await run_problem_graph(
                difficulty,
                category,
                trigger=Trigger.BATCH,
                purpose=ProblemPurpose.NORMAL,
            )
        if problem is not None:
            return True
    return False


async def fetch_demands() -> list[ProblemDemand] | None:
    """Spring 재고를 받는다. 재시도해도 실패하면 None."""
    for attempt, delay in enumerate((*DEMAND_RETRY_DELAYS_S, None), start=1):
        try:
            return await get_problem_demands()
        except Exception as error:
            logger.warning("Spring 재고 조회 실패 (%d번째): %r", attempt, error)
            if delay is None:
                return None
            await asyncio.sleep(delay)
    return None


async def load_pending(purpose: ProblemPurpose) -> dict[ProblemKey, int]:
    """배치로 만들고 아직 보내지 않은 문제 수."""
    async with session_scope() as session:
        return await GeneratedProblemRepository(session).count_pending(purpose)


async def generate_daily() -> None:
    """미전송 데일리가 DAILY_COUNT개보다 적으면 새로 만든다."""
    pending = sum((await load_pending(ProblemPurpose.DAILY)).values())
    if pending >= DAILY_COUNT:
        logger.info("미전송 데일리 %d개가 남아 있어 새로 만들지 않음", pending)
        return

    problems = await generate_daily_problems(Trigger.BATCH)
    logger.info("데일리 생성 끝: %d/%d개", len(problems), DAILY_COUNT)


async def load_battle_pending() -> int:
    """배치로 만들고 아직 보내지 않은 배틀 문제 수."""
    async with session_scope() as session:
        return await BattleProblemRepository(session).count_pending()


async def generate_battle() -> None:
    """
    배틀 문제를 하나 만든다. 폐기되면 MAX_ATTEMPTS번까지 다시 만든다.

    Spring은 하루에 한 문제를 받아 그날의 배틀로 쓴다. 전송에 실패해 남은
    문제가 있으면 그 문제가 다음 날 나가므로 새로 만들지 않는다.
    """
    pending = await load_battle_pending()
    if pending:
        logger.info("미전송 배틀 %d개가 남아 있어 새로 만들지 않음", pending)
        return

    for attempt in range(1, MAX_ATTEMPTS + 1):
        problem = await run_battle_graph(trigger=Trigger.BATCH)
        if problem is not None:
            logger.info(
                "배틀 생성 끝: %s (%d번째 시도)", problem.problem_title, attempt
            )
            return
    logger.warning("배틀 문제를 %d번 시도했지만 만들지 못함", MAX_ATTEMPTS)


async def generate_normal() -> None:
    """조합별 재고가 TARGET_STOCK이 되도록 일반 문제를 만든다."""
    demands = await fetch_demands()
    pending = await load_pending(ProblemPurpose.NORMAL)
    plan = plan_generation(demands or [], pending)
    slots = interleave(plan)

    if demands is None:
        # 조용히 넘어가면 Spring 재고가 충분한 날에도 수백 개를 만든 걸 모른다.
        logger.warning(
            "Spring 재고를 받지 못해 버퍼 재고만 보고 생성함: 조합 %d개, 문제 %d개",
            len(plan),
            len(slots),
        )
    if not slots:
        logger.info("모든 조합의 재고가 충분해 일반 문제를 만들지 않음")
        return

    concurrency = get_settings().batch_concurrency
    logger.info(
        "일반 문제 생성 시작: 조합 %d개, 문제 %d개, 동시 %d개",
        len(plan),
        len(slots),
        concurrency,
    )
    semaphore = asyncio.Semaphore(concurrency)
    started = time.monotonic()
    results = await asyncio.gather(*(fill_slot(key, semaphore) for key in slots))

    missed = Counter(
        key for key, filled in zip(slots, results, strict=True) if not filled
    )
    logger.info(
        "일반 문제 생성 끝: %d/%d개, %.0f초",
        len(slots) - missed.total(),
        len(slots),
        time.monotonic() - started,
    )
    if missed:
        logger.warning(
            "채우지 못한 조합: %s",
            ", ".join(f"{_label(key)}×{n}" for key, n in missed.most_common()),
        )


# 이름은 수동 실행(--only)에서 고를 때 쓴다. 순서가 곧 실행 순서다.
# 데일리·배틀은 Spring 조회가 필요 없고 금방 끝나서 일반 문제보다 먼저 돌린다.
GENERATE_STEPS: dict[str, Step] = {
    "daily": generate_daily,
    "battle": generate_battle,
    "normal": generate_normal,
}

# 배틀은 백엔드가 아직 받지 않아(401) 자동 배치에서 제외한다.
# 수동으로는 --only battle로 돌릴 수 있다.
AUTO_GENERATE_STEPS = ("daily", "normal")


async def run_generate(only: str | None = None) -> None:
    """
    01:00 생성 배치. 스케줄러가 부른다.

    예외를 올리지 않는다. 스케줄러는 결과를 보지 않으므로 로그로 남긴다.
    한 단계가 실패해도 다음 단계는 돈다.

    Parameters:
            only (str | None): 이 단계만 돌린다. 수동 실행에서 쓴다.
            없으면 AUTO_GENERATE_STEPS에 있는 단계만 돈다.
    """
    names = AUTO_GENERATE_STEPS if only is None else (only,)
    steps = [step for name, step in GENERATE_STEPS.items() if name in names]
    try:
        async with try_advisory_lock(LockKey.BATCH_GENERATE) as acquired:
            if not acquired:
                logger.info("다른 인스턴스가 생성 배치를 실행 중이라 건너뜀")
                return

            for step in steps:
                try:
                    await step()
                except Exception:
                    logger.exception("생성 배치 단계 실패: %s", step.__name__)
    except Exception:
        # 락을 잡는 DB 연결부터 실패한 경우
        logger.exception("생성 배치를 시작하지 못함")
