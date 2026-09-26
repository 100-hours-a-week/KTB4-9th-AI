import logging
import warnings

from fastapi.testclient import TestClient

from src.core.config import get_settings
from src.core.logging import setup_logging
from src.main import app


def test_setup_logging_attaches_a_single_console_handler() -> None:
    setup_logging()
    setup_logging()  # 두 번 불러도 핸들러가 쌓이지 않아야 한다
    root = logging.getLogger()

    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0], logging.StreamHandler)


def test_setup_logging_follows_log_level() -> None:
    setup_logging()

    assert (
        logging.getLogger().level
        == logging.getLevelNamesMapping()[get_settings().log_level.upper()]
    )


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


def test_gemini_client_logger_is_quieted() -> None:
    setup_logging()

    # LLM 호출마다 "AFC is enabled" INFO가 찍혀 배치 로그가 묻히는 것을 막는다
    assert logging.getLogger("google_genai").level == logging.WARNING


def warn_like_a_library(message: str) -> None:
    """실제 라이브러리 경고처럼 매번 같은 위치에서 낸다."""
    warnings.warn(message, UserWarning, stacklevel=1)


def test_repeated_library_warning_is_logged_once(capsys) -> None:
    """warnings 모듈이 매번 내보내도(기록이 초기화되는 상황) 로그에는 한 번만 남는다."""
    logging.captureWarnings(False)  # 앞선 테스트가 남긴 전역 상태를 비운다
    setup_logging()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            for _ in range(3):
                warn_like_a_library("temperature will be ignored")
            warn_like_a_library("다른 경고")
    finally:
        logging.captureWarnings(False)

    out = capsys.readouterr().out
    assert out.count("temperature will be ignored") == 1
    assert out.count("다른 경고") == 1
    assert "WARNING py.warnings" in out  # 우리 로그 형식으로 나온다
