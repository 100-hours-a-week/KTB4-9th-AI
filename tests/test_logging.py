import logging

from fastapi.testclient import TestClient

from src.api import app
from src.core.logging import setup_logging
from src.shared.config import get_settings


def test_setup_logging_attaches_a_single_console_handler() -> None:
    setup_logging()
    setup_logging()  # 두 번 불러도 핸들러가 쌓이지 않아야 한다
    root = logging.getLogger()

    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0], logging.StreamHandler)


def test_setup_logging_follows_log_level() -> None:
    setup_logging()

    assert logging.getLogger().level == logging.getLevelNamesMapping()[
        get_settings().log_level.upper()
    ]


def test_http_client_loggers_are_quieted() -> None:
    setup_logging()

    # 요청마다 INFO 한 줄씩 찍혀 우리 로그가 묻히는 것을 막는다
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING


def test_sqlalchemy_logger_follows_db_echo() -> None:
    setup_logging()
    expected = logging.INFO if get_settings().db_echo else logging.WARNING

    assert logging.getLogger("sqlalchemy.engine").level == expected


def test_lifespan_configures_logging_and_logs_startup(capsys) -> None:
    with TestClient(app) as client:
        client.get("/health")

    assert "애플리케이션 시작" in capsys.readouterr().out
