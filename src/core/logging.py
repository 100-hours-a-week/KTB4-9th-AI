import logging
from logging.config import dictConfig

from src.core.config import get_settings


class OncePerMessage(logging.Filter):
    """같은 메시지는 프로세스에서 한 번만 통과시킨다.

    라이브러리 경고(warnings.warn)는 LLM 호출마다 반복된다. warnings 모듈의
    "once"로는 못 막는다. langchain·langgraph가 catch_warnings를 쓸 때마다
    이미 보여 준 경고 기록이 초기화되기 때문이다.
    """

    def __init__(self) -> None:
        super().__init__()
        self._seen: set[str] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if message in self._seen:
            return False
        self._seen.add(message)
        return True


def setup_logging() -> None:
    settings = get_settings()
    level = settings.log_level.upper()

    # warnings.warn을 stderr 대신 로그로 보내 시각·형식을 맞추고 중복을 거른다
    logging.captureWarnings(True)

    dictConfig(
        {
            "version": 1,
            # uvicorn이 자기 로거를 먼저 잡으므로 넘겨받아 다시 설정한다
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                    "datefmt": "%H:%M:%S",
                }
            },
            "filters": {"once_per_message": {"()": OncePerMessage}},
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                    "formatter": "default",
                }
            },
            "root": {"level": level, "handlers": ["console"]},
            "loggers": {
                "uvicorn.access": {"level": level, "propagate": True},
                "uvicorn.error": {"level": level, "propagate": True},
                # db_echo=True일 때만 쿼리를 보여 준다
                "sqlalchemy.engine": {
                    "level": "INFO" if settings.db_echo else "WARNING",
                    "propagate": True,
                },
                # 요청마다 INFO 한 줄씩 찍혀 로그가 묻히는 것을 막는다
                "httpx": {"level": "WARNING", "propagate": True},
                "httpx2": {"level": "WARNING", "propagate": True},
                "httpcore": {"level": "WARNING", "propagate": True},
                # LLM 호출마다 "AFC is enabled" INFO가 찍힌다
                "google_genai": {"level": "WARNING", "propagate": True},
                # 예: 모델이 temperature를 무시한다는 경고가 호출마다 반복된다
                "py.warnings": {
                    "level": "WARNING",
                    "filters": ["once_per_message"],
                    "propagate": True,
                },
            },
        }
    )
