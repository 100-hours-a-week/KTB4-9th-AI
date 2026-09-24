from logging.config import dictConfig

from src.core.config import get_settings


def setup_logging() -> None:
    settings = get_settings()
    level = settings.log_level.upper()

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
            },
        }
    )
