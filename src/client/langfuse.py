"""Langfuse 클라이언트 및 LangChain/LangGraph 콜백 핸들러.

LLM 관측성, 노드별 상세 실행 트레이스, 토큰/비용 모니터링, 프롬프트 관리를 담당한다.
키(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY)가 설정되지 않은 환경에서는
조용히 비활성화되어 로컬 개발 및 유닛 테스트가 외부 연결 없이 정상 동작한다.
"""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from typing import Any

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from src.core.config import get_settings

logger = logging.getLogger(__name__)


def is_langfuse_enabled() -> bool:
    """Langfuse API 키가 설정되어 활성화 가능한지 확인한다."""
    settings = get_settings()
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


def _sync_env_with_settings() -> None:
    """Langfuse SDK 내부에서 os.environ을 참조할 수 있도록 동기화한다."""
    settings = get_settings()
    if settings.langfuse_public_key:
        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
    if settings.langfuse_secret_key:
        os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
    if settings.langfuse_host:
        os.environ["LANGFUSE_HOST"] = settings.langfuse_host


@lru_cache
def get_langfuse_client() -> Langfuse | None:
    """Langfuse 클라이언트를 반환한다. 키가 없으면 None."""
    if not is_langfuse_enabled():
        return None
    settings = get_settings()
    _sync_env_with_settings()
    try:
        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception as error:
        logger.warning("Langfuse 클라이언트 초기화 실패 (비활성화): %s", error)
        return None


def get_langfuse_callback_handler(
    *,
    trace_context: Any = None,
) -> CallbackHandler | None:
    """LangGraph 및 LangChain 실행에 주입할 콜백 핸들러를 반환한다.

    키가 설정되지 않았으면 None을 반환하여 불필요한 네트워크 호출을 방지한다.
    """
    if not is_langfuse_enabled():
        return None
    settings = get_settings()
    _sync_env_with_settings()
    try:
        # Langfuse 클라이언트를 먼저 준비한다
        get_langfuse_client()
        return CallbackHandler(
            public_key=settings.langfuse_public_key,
            trace_context=trace_context,
        )
    except Exception as error:
        logger.warning("Langfuse CallbackHandler 생성 실패: %s", error)
        return None


@contextmanager
def langfuse_observation(name: str, **kwargs: Any) -> Iterator[Any]:
    """Langfuse 관측성 컨텍스트 매니저.

    Langfuse가 활성화된 경우 start_as_current_observation 컨텍스트를 시작하고,
    비활성화된 경우 nullcontext로 아무 동작도 하지 않는다.
    """
    client = get_langfuse_client()
    if client is not None:
        try:
            with client.start_as_current_observation(name=name, **kwargs) as obs:
                yield obs
            return
        except Exception as error:
            logger.warning("Langfuse 관측성 컨텍스트 에러 (계속 진행): %s", error)
            yield None
            return

    with nullcontext() as ctx:
        yield ctx


def flush_langfuse() -> None:
    """남아 있는 Langfuse 트레이스 이벤트를 서버로 플러시한다."""
    client = get_langfuse_client()
    if client is not None:
        try:
            client.flush()
        except Exception as error:
            logger.warning("Langfuse 플러시 실패: %s", error)
