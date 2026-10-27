from unittest.mock import MagicMock

from src.client.langfuse import (
    flush_langfuse,
    get_langfuse_callback_handler,
    get_langfuse_client,
    is_langfuse_enabled,
    langfuse_observation,
)
from src.core.config import Settings


def test_is_langfuse_enabled_false_by_default(monkeypatch):
    monkeypatch.setattr(
        "src.client.langfuse.get_settings",
        lambda: Settings(langfuse_public_key=None, langfuse_secret_key=None),
    )
    get_langfuse_client.cache_clear()
    assert is_langfuse_enabled() is False


def test_is_langfuse_enabled_true_when_keys_present(monkeypatch):
    monkeypatch.setattr(
        "src.client.langfuse.get_settings",
        lambda: Settings(
            langfuse_public_key="pk-test",
            langfuse_secret_key="sk-test",
        ),
    )
    get_langfuse_client.cache_clear()
    assert is_langfuse_enabled() is True


def test_get_langfuse_client_none_when_disabled(monkeypatch):
    monkeypatch.setattr(
        "src.client.langfuse.get_settings",
        lambda: Settings(langfuse_public_key=None, langfuse_secret_key=None),
    )
    get_langfuse_client.cache_clear()
    assert get_langfuse_client() is None


def test_get_langfuse_callback_handler_none_when_disabled(monkeypatch):
    monkeypatch.setattr(
        "src.client.langfuse.get_settings",
        lambda: Settings(langfuse_public_key=None, langfuse_secret_key=None),
    )
    get_langfuse_client.cache_clear()
    assert get_langfuse_callback_handler() is None


def test_langfuse_observation_yields_none_when_disabled(monkeypatch):
    monkeypatch.setattr(
        "src.client.langfuse.get_settings",
        lambda: Settings(langfuse_public_key=None, langfuse_secret_key=None),
    )
    get_langfuse_client.cache_clear()
    executed = False
    with langfuse_observation(name="test_obs") as obs:
        executed = True
        assert obs is None
    assert executed is True


def test_langfuse_observation_uses_client_when_enabled(monkeypatch):
    mock_client = MagicMock()
    mock_ctx = MagicMock()
    mock_client.start_as_current_observation.return_value.__enter__.return_value = (
        mock_ctx
    )
    monkeypatch.setattr(
        "src.client.langfuse.get_settings",
        lambda: Settings(
            langfuse_public_key="pk-test",
            langfuse_secret_key="sk-test",
        ),
    )
    monkeypatch.setattr("src.client.langfuse.get_langfuse_client", lambda: mock_client)

    with langfuse_observation(name="test_obs", as_type="chain") as obs:
        assert obs == mock_ctx
    mock_client.start_as_current_observation.assert_called_once_with(
        name="test_obs", as_type="chain"
    )


def test_flush_langfuse_safe_when_disabled():
    get_langfuse_client.cache_clear()
    flush_langfuse()  # 에러 없이 완료


def test_flush_langfuse_calls_client_flush(monkeypatch):
    mock_client = MagicMock()
    monkeypatch.setattr("src.client.langfuse.get_langfuse_client", lambda: mock_client)
    flush_langfuse()
    mock_client.flush.assert_called_once()
