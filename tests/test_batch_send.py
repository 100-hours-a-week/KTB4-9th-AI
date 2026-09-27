import contextlib
import logging
import uuid

import httpx
import pytest

from src.batch import send as module
from src.batch.send import SEND_CHUNK_SIZE, SEND_RETRY_DELAYS_S, is_rejection, run_send
from src.core.enums import Category, Difficulty, ProblemPurpose
from src.problem.daily import DAILY_COUNT
from src.schema.problem import Problem

NORMAL = ProblemPurpose.NORMAL
DAILY = ProblemPurpose.DAILY


def make_problem(title: str) -> Problem:
    return Problem(
        problem_title=title,
        problem_content="합이 M인 쌍의 수를 구하라",
        input_format="N M",
        output_format="쌍의 수",
        requested_difficulty=Difficulty.LV2,
        difficulty=Difficulty.LV2,
        category=Category.HASH_TABLE,
        category_select_reason="해시맵으로 푼다",
        input_constraints=[],
        execution_limits=[],
        problem_examples=[],
    )


def status_error(status: int, text: str = "") -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://backend.test/problems")
    response = httpx.Response(status, text=text, request=request)
    return httpx.HTTPStatusError(f"{status}", request=request, response=response)


class FakeBuffer:
    """generated_problems 대역. 넣은 순서가 곧 오래된 순서다."""

    def __init__(self) -> None:
        self.rows: list[tuple[uuid.UUID, ProblemPurpose, Problem]] = []
        self.sent: list[uuid.UUID] = []
        self.mark_error: Exception | None = None

    def add(self, purpose: ProblemPurpose, count: int) -> list[str]:
        """문제를 넣고 제목을 돌려준다. 제목으로 요청 내용을 확인한다."""
        titles = []
        for _ in range(count):
            title = f"{purpose.value}-{len(self.rows)}"
            self.rows.append((uuid.uuid4(), purpose, make_problem(title)))
            titles.append(title)
        return titles

    def id_of(self, title: str) -> uuid.UUID:
        return next(i for i, _, p in self.rows if p.problem_title == title)

    def sent_titles(self) -> list[str]:
        by_id = {i: p.problem_title for i, _, p in self.rows}
        return [by_id[i] for i in self.sent]

    async def load_chunk(self, purpose, limit, exclude_ids=()):
        pending = [
            (i, p)
            for i, pur, p in self.rows
            if pur == purpose and i not in self.sent and i not in exclude_ids
        ]
        return pending[:limit]

    async def mark_sent(self, problem_ids):
        if self.mark_error:
            raise self.mark_error
        self.sent.extend(problem_ids)


class FakeSpring:
    """Spring 대역. 요청마다 (경로, 제목 목록)을 남긴다."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, list[str]]] = []
        self.failures: list[Exception] = []  # 앞에서부터 꺼내 던진다
        self.reject_titles: set[str] = set()  # 이 제목이 든 요청은 400
        self.down_after: int | None = None  # 이 수를 넘는 요청부터는 계속 503

    async def save_problems(self, problems):
        self.handle("/problems", problems)

    async def save_daily_problems(self, problems):
        self.handle("/daily-problems", problems)

    def handle(self, path: str, problems: list[Problem]) -> None:
        titles = [p.problem_title for p in problems]
        self.requests.append((path, titles))
        if self.down_after is not None and len(self.requests) > self.down_after:
            raise status_error(503)
        if self.failures:
            raise self.failures.pop(0)
        if self.reject_titles & set(titles):
            raise status_error(400, "problemContent는 비어 있을 수 없습니다")

    def sizes(self, path: str = "/problems") -> list[int]:
        return [len(titles) for p, titles in self.requests if p == path]


@pytest.fixture
def buffer(monkeypatch: pytest.MonkeyPatch) -> FakeBuffer:
    fake = FakeBuffer()
    monkeypatch.setattr(module, "load_chunk", fake.load_chunk)
    monkeypatch.setattr(module, "mark_sent", fake.mark_sent)
    return fake


@pytest.fixture
def spring(monkeypatch: pytest.MonkeyPatch) -> FakeSpring:
    fake = FakeSpring()
    monkeypatch.setattr(module, "save_problems", fake.save_problems)
    monkeypatch.setattr(module, "save_daily_problems", fake.save_daily_problems)
    monkeypatch.setattr(module, "SEND_RETRY_DELAYS_S", (0,) * len(SEND_RETRY_DELAYS_S))

    @contextlib.asynccontextmanager
    async def always_acquired(key):
        yield True

    monkeypatch.setattr(module, "try_advisory_lock", always_acquired)
    return fake


# ── 오류 구분 ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (status_error(400), True),
        (status_error(404), True),
        (status_error(422), True),
        (status_error(408), False),  # 기다리면 되는 4xx
        (status_error(429), False),
        (status_error(500), False),
        (status_error(503), False),
        (httpx.ConnectError("connection refused"), False),
        (httpx.ReadTimeout("timed out"), False),
    ],
)
def test_only_content_rejections_count_as_rejection(error, expected) -> None:
    assert is_rejection(error) is expected


# ── 데일리 ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_daily_sends_the_oldest_set_in_one_request(buffer, spring) -> None:
    titles = buffer.add(DAILY, DAILY_COUNT + 2)

    await run_send()

    assert spring.requests == [("/daily-problems", titles[:DAILY_COUNT])]
    assert buffer.sent_titles() == titles[:DAILY_COUNT]


@pytest.mark.asyncio
async def test_short_daily_is_sent_as_it_is(buffer, spring, caplog) -> None:
    titles = buffer.add(DAILY, 3)

    with caplog.at_level(logging.WARNING):
        await run_send()

    assert spring.requests == [("/daily-problems", titles)]
    assert "3/5" in caplog.text


@pytest.mark.asyncio
async def test_missing_daily_is_warned_and_normal_still_goes(
    buffer, spring, caplog
) -> None:
    titles = buffer.add(NORMAL, 2)

    with caplog.at_level(logging.WARNING):
        await run_send()

    assert spring.requests == [("/problems", titles)]
    assert "보낼 데일리가 없음" in caplog.text


@pytest.mark.asyncio
async def test_rejected_daily_stays_whole_and_normal_still_goes(
    buffer, spring, caplog
) -> None:
    """데일리는 세트라 한 건씩 쪼개 보내지 않는다."""
    daily = buffer.add(DAILY, DAILY_COUNT)
    normal = buffer.add(NORMAL, 2)
    spring.reject_titles = {daily[0]}

    with caplog.at_level(logging.ERROR):
        await run_send()

    assert spring.sizes("/daily-problems") == [DAILY_COUNT]
    assert buffer.sent_titles() == normal
    # 사유와 id가 한 줄에 남고, traceback만 남는 단계 실패로 새지 않는다
    assert "데일리 전송 실패(거부)" in caplog.text
    assert "problemContent는 비어 있을 수 없습니다" in caplog.text
    assert str(buffer.id_of(daily[0])) in caplog.text
    assert "전송 배치 단계 실패" not in caplog.text


@pytest.mark.asyncio
async def test_daily_outage_is_retried_then_left_whole(buffer, spring, caplog) -> None:
    buffer.add(DAILY, DAILY_COUNT)
    spring.failures = [status_error(503)] * (1 + len(SEND_RETRY_DELAYS_S))

    with caplog.at_level(logging.ERROR):
        await run_send()

    assert spring.sizes("/daily-problems") == [DAILY_COUNT] * (
        1 + len(SEND_RETRY_DELAYS_S)
    )
    assert buffer.sent == []
    assert "데일리 전송 실패(장애 지속)" in caplog.text
    assert "전송 배치 단계 실패" not in caplog.text


# ── 일반 문제 ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_normal_is_sent_in_chunks_until_empty(buffer, spring) -> None:
    titles = buffer.add(NORMAL, SEND_CHUNK_SIZE * 2 + 5)

    await run_send()

    assert spring.sizes() == [SEND_CHUNK_SIZE, SEND_CHUNK_SIZE, 5]
    assert buffer.sent_titles() == titles


@pytest.mark.asyncio
async def test_transient_failure_is_retried_then_sent(buffer, spring) -> None:
    titles = buffer.add(NORMAL, 3)
    spring.failures = [status_error(503), httpx.ConnectError("refused")]

    await run_send()

    assert spring.sizes() == [3, 3, 3]
    assert buffer.sent_titles() == titles


@pytest.mark.asyncio
async def test_too_many_requests_is_retried_not_split(buffer, spring) -> None:
    """429를 거부로 보면 멀쩡한 청크를 한 건씩 쪼개 보내게 된다."""
    buffer.add(NORMAL, 3)
    spring.failures = [status_error(429)]

    await run_send()

    assert spring.sizes() == [3, 3]


@pytest.mark.asyncio
async def test_persistent_failure_stops_for_the_day(buffer, spring, caplog) -> None:
    buffer.add(NORMAL, SEND_CHUNK_SIZE + 5)
    spring.failures = [status_error(503)] * (1 + len(SEND_RETRY_DELAYS_S))

    with caplog.at_level(logging.ERROR):
        await run_send()

    assert spring.sizes() == [SEND_CHUNK_SIZE] * (1 + len(SEND_RETRY_DELAYS_S))
    assert buffer.sent == []
    assert "다음 날" in caplog.text


@pytest.mark.asyncio
async def test_rejected_chunk_is_resent_one_by_one(buffer, spring, caplog) -> None:
    first, bad, last = buffer.add(NORMAL, 3)
    spring.reject_titles = {bad}

    with caplog.at_level(logging.ERROR):
        await run_send()

    assert [titles for _, titles in spring.requests] == [
        [first, bad, last],
        [first],
        [bad],
        [last],
    ]
    assert buffer.sent_titles() == [first, last]
    assert str(buffer.id_of(bad)) in caplog.text
    assert "problemContent는 비어 있을 수 없습니다" in caplog.text


@pytest.mark.asyncio
async def test_rejected_row_does_not_block_later_chunks(buffer, spring) -> None:
    """가장 오래된 행이 거부돼도 뒤의 문제는 모두 나가야 한다."""
    titles = buffer.add(NORMAL, SEND_CHUNK_SIZE + 5)
    bad = titles[0]
    spring.reject_titles = {bad}

    await run_send()

    assert buffer.sent_titles() == titles[1:]
    tried_bad = [t for _, t in spring.requests if bad in t]
    assert len(tried_bad) == 2  # 청크 한 번, 한 건 한 번. 같은 실행에서 다시 안 보낸다


@pytest.mark.asyncio
async def test_outage_while_splitting_keeps_what_was_sent(buffer, spring) -> None:
    """한 건씩 보내다 장애가 나도, 앞서 보낸 건은 표시돼 다음 날 다시 안 나간다."""
    first, bad, last = buffer.add(NORMAL, 3)
    spring.reject_titles = {bad}
    spring.down_after = 2  # 청크 거부, 첫 건 성공. 그 뒤로는 계속 장애

    await run_send()

    assert buffer.sent_titles() == [first]


@pytest.mark.asyncio
async def test_mark_failure_after_save_is_logged_and_stops(
    buffer, spring, caplog
) -> None:
    titles = buffer.add(NORMAL, SEND_CHUNK_SIZE + 5)
    buffer.mark_error = OSError("DB 연결 끊김")

    with caplog.at_level(logging.ERROR):
        await run_send()

    assert spring.sizes() == [SEND_CHUNK_SIZE]  # 같은 청크를 또 보내지 않는다
    assert "중복 전송될 수 있음" in caplog.text
    assert str(buffer.id_of(titles[0])) in caplog.text


# ── 실행 ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_another_instance_holding_the_lock_sends_nothing(
    buffer, spring, monkeypatch
) -> None:
    @contextlib.asynccontextmanager
    async def held(key):
        yield False

    monkeypatch.setattr(module, "try_advisory_lock", held)
    buffer.add(DAILY, DAILY_COUNT)
    buffer.add(NORMAL, 3)

    await run_send()

    assert spring.requests == []


@pytest.mark.asyncio
async def test_lock_error_is_logged_not_raised(
    buffer, spring, monkeypatch, caplog
) -> None:
    @contextlib.asynccontextmanager
    async def broken(key):
        raise OSError("DB 연결 거부")
        yield  # pragma: no cover

    monkeypatch.setattr(module, "try_advisory_lock", broken)

    with caplog.at_level(logging.ERROR):
        await run_send()

    assert "전송 배치를 시작하지 못함" in caplog.text
